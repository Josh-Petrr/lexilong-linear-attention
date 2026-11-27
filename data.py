"""Dataset loading + tokenisation (cached to disk so each training run starts instantly).

Two training corpora:
  * "imdb"     - IMDB movie reviews (25k train / 25k test). Used for the controlled
                 baseline-vs-improved benchmark (Phase 3).
  * "feedback" - IMDB train + a 50k sample of Yelp business reviews (Yelp Polarity).
                 Yelp reviews are customer feedback about service, staff, delivery,
                 refunds..., i.e. the domain of the Phase 4 demo. Evaluated on both the
                 IMDB test set and a 10k sample of the Yelp test set.
"""
import os

import torch
from torch.utils.data import ConcatDataset, TensorDataset

TOKENIZER_NAME = "bert-base-uncased"
CACHE_DIR = os.path.join(os.path.dirname(__file__), "data_cache")


def get_tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(TOKENIZER_NAME)


def _clean(t):
    return t.replace("<br />", " ").replace("\\n", " ").replace('\\"', '"')


def _tokenize_split(tokenizer, texts, labels, max_len):
    texts = [_clean(t) for t in texts]
    enc = tokenizer(texts, truncation=True, max_length=max_len, padding="max_length", return_tensors="pt")
    # Untruncated lengths, to report how long the documents really are.
    lens = torch.tensor([len(x) for x in tokenizer(texts, truncation=False)["input_ids"]])
    return dict(ids=enc["input_ids"], mask=enc["attention_mask"].bool(), labels=torch.tensor(labels), lens=lens)


def _cached(name, max_len, builder):
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f"{name}_{max_len}.pt")
    if os.path.exists(path):
        return torch.load(path)
    blob = builder()
    torch.save(blob, path)
    return blob


def _imdb_blob(tokenizer, max_len):
    def build():
        from datasets import load_dataset

        ds = load_dataset("stanfordnlp/imdb")
        return {s: _tokenize_split(tokenizer, ds[s]["text"], ds[s]["label"], max_len) for s in ("train", "test")}

    return _cached("imdb", max_len, build)


def _yelp_blob(tokenizer, max_len, n_train=50000, n_test=10000, seed=0):
    def build():
        from datasets import load_dataset

        ds = load_dataset("fancyzhx/yelp_polarity")
        out = {}
        for split, n in (("train", n_train), ("test", n_test)):
            sub = ds[split].shuffle(seed=seed).select(range(n))
            out[split] = _tokenize_split(tokenizer, sub["text"], sub["label"], max_len)
        return out

    return _cached(f"yelp{n_train // 1000}k", max_len, build)


def _tds(d, idx=None):
    if idx is None:
        return TensorDataset(d["ids"], d["mask"], d["labels"])
    return TensorDataset(d["ids"][idx], d["mask"][idx], d["labels"][idx])


def _split_train_val(d, val_size, seed):
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(d["labels"]), generator=g)
    return _tds(d, perm[val_size:]), _tds(d, perm[:val_size])


def _len_stats(d, max_len):
    lens = d["lens"].float()
    return dict(len_mean=lens.mean().item(), len_median=lens.median().item(),
                len_p90=lens.quantile(0.9).item(), frac_truncated=(lens > max_len).float().mean().item())


def load_data(name="imdb", max_len=512, val_size=2500, seed=42):
    """Returns (train_ds, val_ds, test_sets: {name: ds}, tokenizer, stats: {name: length stats})."""
    tokenizer = get_tokenizer()
    imdb = _imdb_blob(tokenizer, max_len)
    tr, va = _split_train_val(imdb["train"], val_size, seed)
    tests = {"imdb": _tds(imdb["test"])}
    stats = {"imdb": _len_stats(imdb["test"], max_len)}
    if name == "feedback":
        yelp = _yelp_blob(tokenizer, max_len)
        ytr, yva = _split_train_val(yelp["train"], val_size, seed)
        tr, va = ConcatDataset([tr, ytr]), ConcatDataset([va, yva])
        tests["yelp"] = _tds(yelp["test"])
        stats["yelp"] = _len_stats(yelp["test"], max_len)
    elif name != "imdb":
        raise ValueError(f"unknown dataset {name}")
    return tr, va, tests, tokenizer, stats
