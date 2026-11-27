"""
LexiLong - Long Customer Feedback Analyzer (Gradio demo).

    python app.py            # opens http://127.0.0.1:7860

Tab 1  Analyze feedback : paste a long review / support ticket / complaint. Every trained model
                          classifies it; the app shows sentiment, confidence, latency and memory.
Tab 2  Long-document stress test : feed synthetic documents of 1k-16k tokens through the baseline
                          and the linear-attention model and watch the baseline slow down or OOM.
Tab 3  Benchmark results : the offline plots produced by benchmark.py.
"""
import os
import time

import gradio as gr
import pandas as pd
import torch

from data import get_tokenizer
from models import LexiLongClassifier

ROOT = os.path.dirname(os.path.abspath(__file__))
CKPT_DIR = os.path.join(ROOT, "checkpoints")
RESULTS = os.path.join(ROOT, "results")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NAMES = {"softmax": "Softmax (baseline)", "linear": "Linear attention (LexiLong)", "linformer": "Linformer"}

tokenizer = get_tokenizer()
MODELS = {}
for attn in ("linear", "softmax", "linformer"):
    # Prefer the deployment models trained on IMDB + Yelp customer reviews; fall back to IMDB-only.
    for name in (f"feedback_{attn}.pt", f"{attn}.pt"):
        path = os.path.join(CKPT_DIR, name)
        if os.path.exists(path):
            MODELS[attn] = LexiLongClassifier.from_checkpoint(path).to(DEVICE).eval()
            print(f"loaded {name}")
            break
if not MODELS:
    raise SystemExit("No checkpoints found - run train.py first.")


def _timed_forward(model, ids, mask, repeats=3):
    """Median wall-clock latency (ms) of a forward pass, plus peak memory (MB)."""
    times = []
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        model(ids, mask)  # warm-up
        for _ in range(repeats):
            if DEVICE.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            logits = model(ids, mask)
            if DEVICE.type == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)
    mem = torch.cuda.max_memory_allocated() / 2**20 if DEVICE.type == "cuda" else float("nan")
    return logits, sorted(times)[len(times) // 2], mem


def analyze(text, max_tokens):
    text = (text or "").strip()
    if not text:
        return {}, pd.DataFrame(), "Paste some feedback first."
    full_len = len(tokenizer(text, truncation=False)["input_ids"])
    rows, primary = [], None
    for attn, model in MODELS.items():
        limit = int(max_tokens)
        if attn == "linformer":
            # Linformer's projection matrices are tied to the training length.
            limit = min(limit, model.config["max_len"])
        enc = tokenizer(text, truncation=True, max_length=limit, return_tensors="pt")
        ids = enc["input_ids"].to(DEVICE)
        mask = enc["attention_mask"].bool().to(DEVICE)
        try:
            logits, ms, mem = _timed_forward(model, ids, mask)
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            rows.append({"Model": NAMES[attn], "Sentiment": "OOM", "Confidence": None,
                         "Tokens read": ids.shape[1], "Latency (ms)": None, "Peak GPU mem (MB)": None})
            continue
        probs = torch.softmax(logits.float(), -1)[0].cpu()
        label = "Positive" if probs[1] > probs[0] else "Negative"
        rows.append({"Model": NAMES[attn], "Sentiment": label, "Confidence": round(float(probs.max()), 3),
                     "Tokens read": ids.shape[1], "Latency (ms)": round(ms, 2),
                     "Peak GPU mem (MB)": round(mem, 1)})
        if primary is None:
            primary = {"Positive": float(probs[1]), "Negative": float(probs[0])}
    note = (f"Document length: **{full_len} tokens**. "
            f"{'It was truncated to ' + str(int(max_tokens)) + ' tokens.' if full_len > max_tokens else 'Read in full.'} "
            "Models were trained on up to 512 tokens; softmax and linear attention use sinusoidal positions, "
            "so they can read longer inputs. Linformer is capped at its training length. "
            "For a single short document, fixed overheads dominate and all models take a few ms; "
            "the gap opens up as documents get longer (see the stress test).")
    return primary or {}, pd.DataFrame(rows), note


def stress_test(seq_len, batch_size):
    seq_len, batch_size = int(seq_len), int(batch_size)
    rows = []
    for attn in ("softmax", "linear"):
        if attn not in MODELS:
            continue
        ids = torch.randint(1000, tokenizer.vocab_size, (batch_size, seq_len), device=DEVICE)
        mask = torch.ones_like(ids, dtype=torch.bool)
        try:
            _, ms, mem = _timed_forward(MODELS[attn], ids, mask, repeats=3)
            rows.append({"Model": NAMES[attn], "Tokens": seq_len * batch_size, "Latency (ms)": round(ms, 1),
                         "Peak GPU mem (MB)": round(mem, 0),
                         "Throughput (tok/s)": int(seq_len * batch_size / (ms / 1000))})
        except torch.OutOfMemoryError:
            rows.append({"Model": NAMES[attn], "Tokens": seq_len * batch_size, "Latency (ms)": "OOM",
                         "Peak GPU mem (MB)": "OOM", "Throughput (tok/s)": "OOM"})
        finally:
            del ids, mask
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()
    df = pd.DataFrame(rows)
    msg = ""
    if len(rows) == 2 and all(isinstance(r["Latency (ms)"], float) for r in rows):
        msg = f"Linear attention is **{rows[0]['Latency (ms)'] / rows[1]['Latency (ms)']:.1f}x faster** and uses " \
              f"**{rows[0]['Peak GPU mem (MB)'] / rows[1]['Peak GPU mem (MB)']:.1f}x less memory** at this length."
    elif any(r["Latency (ms)"] == "OOM" for r in rows):
        msg = "The softmax baseline **ran out of GPU memory**; linear attention handled the same input."
    if DEVICE.type == "cuda":
        vram = torch.cuda.get_device_properties(0).total_memory / 2**20
        if any(isinstance(r["Peak GPU mem (MB)"], float) and r["Peak GPU mem (MB)"] > vram for r in rows):
            msg += (f" Note: peak memory exceeded the GPU's {vram / 1024:.0f} GB of dedicated VRAM, so the driver "
                    "spilled into much slower system RAM. That is why latency jumps so sharply.")
    return df, msg


EXAMPLE_TICKET = (
    "I have been a customer of your internet service for almost six years and I have never felt the need to write "
    "a complaint until now. Three weeks ago my connection started dropping every evening between 7 and 11 pm, which "
    "is exactly when my family needs it most. I called support on the first day and was told it was a regional "
    "issue that would be fixed within 48 hours. It was not. I called again four days later, waited on hold for an "
    "hour and ten minutes, and the agent asked me to restart my router, something I had already done a dozen times. "
    "A technician visit was scheduled for the following Tuesday between 8 am and 12 pm. I took the morning off work. "
    "Nobody came, nobody called, and when I phoned in the afternoon I was told the appointment had been cancelled "
    "'by the system'. The second appointment was rescheduled twice. When the technician finally arrived he was "
    "polite but admitted he had not been given any notes about my case, so we started again from zero. He replaced "
    "a cable outside the house and said everything looked fine. The drops continued that same night. Meanwhile I am "
    "still being billed the full monthly price, and the 'loyalty discount' I was promised last year quietly "
    "disappeared from my invoice. I asked for a credit for the days without service and was offered five dollars. "
    "Five dollars for three weeks of unusable internet, two missed work mornings and roughly four hours on hold. "
    "Your website says you value long-term customers, but nothing in this experience reflects that. Unless the "
    "problem is fixed this week and my bill is corrected, I will cancel and move to another provider, and I will "
    "make sure my neighbours, who have been asking me about the outages, know exactly why."
)
EXAMPLE_REVIEW = (
    "I went into this film with fairly low expectations, mostly because the trailer made it look like yet another "
    "formulaic thriller, and I came out of the cinema genuinely moved. The first act takes its time, and some viewers "
    "may find it slow, but every quiet scene is doing work: by the time the story turns, you know these characters "
    "well enough that every decision feels earned rather than convenient. The lead performance is extraordinary. "
    "There is a long, nearly wordless sequence in a hospital corridor where the whole history of a marriage plays "
    "out on one face, and the audience around me was completely silent. The supporting cast is just as strong, "
    "especially the younger brother, who could easily have been comic relief but is instead given real depth. "
    "Visually it is beautiful without being showy; the cinematographer uses natural light and long takes, and the "
    "score is restrained, coming in only when it matters. If I had one criticism it would be that a subplot about the "
    "family business is introduced and then mostly forgotten, but it is a small flaw in an otherwise thoughtful, "
    "patient and deeply human film. It is the best thing I have seen this year and I have already recommended it "
    "to everyone I know. Go and see it on a big screen if you can."
)

with gr.Blocks(title="LexiLong - Long Feedback Analyzer") as demo:
    gr.Markdown(
        "# LexiLong: Long Customer Feedback Analyzer\n"
        "Sentiment analysis of **long** reviews, support tickets and complaints with a Transformer whose "
        "O(n²) softmax attention has been replaced by **O(n) linear attention** (Katharopoulos et al., 2020). "
        f"Running on **{torch.cuda.get_device_name() if DEVICE.type == 'cuda' else 'CPU'}** with "
        f"models: {', '.join(NAMES[a] for a in MODELS)}."
    )
    with gr.Tab("Analyze feedback"):
        with gr.Row():
            with gr.Column(scale=3):
                inp = gr.Textbox(lines=14, label="Customer feedback / review / support ticket",
                                 placeholder="Paste a long piece of feedback here...")
                max_tok = gr.Slider(128, 4096, value=1024, step=128, label="Max tokens to read")
                btn = gr.Button("Analyze", variant="primary")
                gr.Examples([[EXAMPLE_TICKET, 1024], [EXAMPLE_REVIEW, 1024]], [inp, max_tok],
                            label="Examples: support complaint, long film review")
            with gr.Column(scale=2):
                label_out = gr.Label(label="Sentiment (primary model: linear attention)", num_top_classes=2)
                note_out = gr.Markdown()
        table_out = gr.Dataframe(label="Side-by-side model comparison", interactive=False)
        btn.click(analyze, [inp, max_tok], [label_out, table_out, note_out])

    with gr.Tab("Long-document stress test"):
        gr.Markdown("Feed a synthetic document of N tokens through the softmax baseline and the "
                    "linear-attention model. Softmax cost grows with N², linear cost grows with N.")
        with gr.Row():
            n_slider = gr.Slider(512, 16384, value=4096, step=512, label="Document length (tokens)")
            b_slider = gr.Slider(1, 16, value=4, step=1, label="Documents per batch")
        run_btn = gr.Button("Run stress test", variant="primary")
        stress_msg = gr.Markdown()
        stress_table = gr.Dataframe(interactive=False)
        run_btn.click(stress_test, [n_slider, b_slider], [stress_table, stress_msg])

    with gr.Tab("Benchmark results"):
        imgs = [os.path.join(RESULTS, f) for f in
                ("latency_infer.png", "memory_infer.png", "latency_train.png", "memory_train.png", "accuracy.png")]
        imgs = [p for p in imgs if os.path.exists(p)]
        if imgs:
            gr.Gallery(imgs, columns=2, height="auto", label="Offline benchmark (benchmark.py / summarize.py)")
        else:
            gr.Markdown("Run `python benchmark.py` and `python summarize.py` to generate the plots.")

if __name__ == "__main__":
    demo.launch()
