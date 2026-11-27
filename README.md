# LexiLong

**Overcoming the quadratic attention bottleneck in Transformers using linear attention for long-document classification**

Standard self-attention compares every token with every other token, so its time and memory grow as **O(N²)** with document length. LexiLong replaces softmax attention with **kernelised linear attention** (Katharopoulos et al., 2020), which costs **O(N)**, and compares it with the standard Transformer and with **Linformer** (Wang et al., 2020). The improved model is deployed as a Gradio app: a **long customer feedback analyzer**.

**Start here: [LexiLong.ipynb](LexiLong.ipynb).** The notebook walks through the whole project (code, training, benchmarks and demo) with an explanation and interpretation of every result. All outputs are saved, so it can be read on GitHub without running it.

## Key results

Measured on an RTX 3050 laptop GPU (6 GB).

| | Softmax (baseline) | **Linear attention (ours)** | Linformer |
|---|---:|---:|---:|
| IMDB test accuracy | 86.52% | 85.77% | 80.20% |
| Yelp test accuracy (models trained on IMDB + 50k Yelp) | 92.04% | **92.19%** | 90.16% |
| Training time per epoch at 512 tokens | 121 s | **36 s** | 38 s |
| Inference at 4,096 tokens (batch 8): latency / memory | 619 ms / 4.25 GB | **37 ms / 0.20 GB** | 40 ms / 0.30 GB |
| Longest document processed (batch 8) | 4,096 tokens | **16,384 tokens** | fixed to its training length |
| Long reviews (> 512 tokens): cut at 512 vs. read in full | 81.9% → 84.9% | 80.9% → **84.4%** | n/a |

## Repository contents

```
LexiLong.ipynb        complete walkthrough notebook (start here)
data.py               IMDB + Yelp download and tokenisation (cached on first run)
models/               attention variants and the shared encoder (used by app.py)
app.py                Gradio demo: Long Customer Feedback Analyzer
checkpoints/          the six trained models
results/              saved metrics, benchmark tables, plots and demo screenshots
requirements.txt      Python dependencies
```

## How to run

```bash
git clone https://github.com/Josh-Petrr/lexilong-linear-attention.git
cd lexilong-linear-attention
pip install -r requirements.txt

jupyter notebook LexiLong.ipynb    # walkthrough; uses the included checkpoints (~10 min, CUDA GPU)
python app.py                      # demo at http://127.0.0.1:7860
```

The datasets are downloaded and tokenised automatically on the first run. Set `RUN_TRAINING = True` in the notebook's setup cell to re-train all models from scratch.

## References
* Vaswani et al. (2017). *Attention Is All You Need.* NeurIPS.
* Katharopoulos et al. (2020). *Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention.* ICML.
* Wang et al. (2020). *Linformer: Self-Attention with Linear Complexity.* arXiv:2006.04768.
* Maas et al. (2011). *Learning Word Vectors for Sentiment Analysis* (IMDB dataset). ACL.
* Zhang, Zhao and LeCun (2015). *Character-level Convolutional Networks for Text Classification* (Yelp Polarity). NeurIPS.
