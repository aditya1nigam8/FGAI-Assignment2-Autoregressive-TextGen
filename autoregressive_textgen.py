# %% [markdown]
# # Autoregressive Word-Level Text Generation: SimpleRNN vs BiRNN vs LSTM vs GRU
# **Dataset:** *Alice's Adventures in Wonderland* (Lewis Carroll), Project Gutenberg #11
# https://www.gutenberg.org/cache/epub/11/pg11.txt

# %%
import os, re, time, json, random, collections, math
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 42
def set_seed(s=SEED):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
set_seed()
torch.set_num_threads(os.cpu_count() or 4)
DEVICE = "cpu"
ROOT = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
OUT = os.path.join(ROOT, "results"); os.makedirs(OUT, exist_ok=True)

# Hyper-parameters (identical for all four models)
SEQ_LEN, EMB, HID, DROP = 10, 128, 256, 0.3
EPOCHS, BATCH, LR, MIN_FREQ = 25, 128, 2e-3, 2
PATIENCE = 4

# %% [markdown]
# ## 1. Dataset: loading, characteristics, preprocessing

# %%
raw = open(os.path.join(ROOT, "data", "alice.txt"), encoding="utf-8").read()
start = raw.index("*** START OF"); start = raw.index("\n", start) + 1
end = raw.index("*** END OF")
text = raw[start:end]
text = re.sub(r"^\s*(CHAPTER [IVXL]+\.?.*)$", " ", text, flags=re.M)   # drop chapter headings
text = text.replace("‘", "'").replace("’", "'").replace("“", '"').replace("”", '"')
text = text.replace("_", "").lower()
# words (incl. internal apostrophes) and punctuation as separate tokens
tokens = re.findall(r"[a-z]+(?:'[a-z]+)*|[.,!?;:\"()-]", text)

cnt = collections.Counter(tokens)
stats = {
    "characters_after_cleaning": len(text),
    "total_tokens": len(tokens),
    "unique_tokens_raw": len(cnt),
    "hapax_legomena": sum(1 for c in cnt.values() if c == 1),
    "top_15": cnt.most_common(15),
}
itos = ["<pad>", "<unk>"] + [w for w, c in cnt.most_common() if c >= MIN_FREQ]
stoi = {w: i for i, w in enumerate(itos)}
V = len(itos)
ids = np.array([stoi.get(t, 1) for t in tokens], dtype=np.int64)
stats["vocab_size_min_freq"] = V
stats["unk_rate"] = float((ids == 1).mean())
print(json.dumps(stats, indent=1))

# token frequency plot
fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
top = cnt.most_common(20)
ax[0].bar([w for w, _ in top], [c for _, c in top], color="#3b6ea5"); ax[0].tick_params(axis="x", rotation=70)
ax[0].set_title("20 most frequent tokens")
fr = np.array(sorted(cnt.values(), reverse=True))
ax[1].loglog(np.arange(1, len(fr) + 1), fr, color="#c0504d"); ax[1].set_title("Zipf plot (rank vs frequency)")
ax[1].set_xlabel("rank"); ax[1].set_ylabel("frequency")
plt.tight_layout(); plt.savefig(os.path.join(OUT, "dataset_characteristics.png"), dpi=140); plt.close()

# %% [markdown]
# ## 2. Input -> target sequence construction
# Sliding window (stride 1): input = `SEQ_LEN` consecutive tokens, target = the next token.
# Contiguous chronological split (85/7.5/7.5 %) so that validation/test text is unseen.

# %%
def make_xy(a):
    n = len(a) - SEQ_LEN
    X = np.stack([a[i:i + SEQ_LEN] for i in range(n)])
    y = a[SEQ_LEN:SEQ_LEN + n]
    return torch.from_numpy(X), torch.from_numpy(y)

n = len(ids); a, b = int(n * 0.85), int(n * 0.925)
Xtr, ytr = make_xy(ids[:a]); Xva, yva = make_xy(ids[a:b]); Xte, yte = make_xy(ids[b:])
print("train/val/test samples:", len(Xtr), len(Xva), len(Xte))
print("example:", [itos[i] for i in Xtr[100]], "->", itos[ytr[100]])
stats.update(train_samples=len(Xtr), val_samples=len(Xva), test_samples=len(Xte))

train_ngrams = set(zip(ids[:a - 1], ids[1:a]))       # train bigrams, for coherence proxy
train_trigrams = set(zip(ids[:a - 2], ids[1:a - 1], ids[2:a]))

# %% [markdown]
# ## 3. Models

# %%
class LM(nn.Module):
    def __init__(self, kind):
        super().__init__()
        self.kind = kind
        self.emb = nn.Embedding(V, EMB, padding_idx=0)
        bi = kind == "BiRNN"
        cell = {"SimpleRNN": nn.RNN, "BiRNN": nn.RNN, "LSTM": nn.LSTM, "GRU": nn.GRU}[kind]
        kw = dict(nonlinearity="tanh") if kind in ("SimpleRNN", "BiRNN") else {}
        self.rnn = cell(EMB, HID, batch_first=True, bidirectional=bi, **kw)
        self.drop = nn.Dropout(DROP)
        self.fc = nn.Linear(HID * (2 if bi else 1), V)
    def forward(self, x):
        h, _ = self.rnn(self.drop(self.emb(x)))
        return self.fc(self.drop(h[:, -1]))   # last time step summarises the context window

def n_params(m): return sum(p.numel() for p in m.parameters())

# %% [markdown]
# ## 4. Training (identical conditions: seed, optimiser, lr, batch, epochs, early stopping)

# %%
@torch.no_grad()
def evaluate(m, X, y):
    m.eval(); tot, c1, c5 = 0.0, 0, 0
    for i in range(0, len(X), 1024):
        xb, yb = X[i:i + 1024], y[i:i + 1024]
        lo = m(xb)
        tot += F.cross_entropy(lo, yb, reduction="sum").item()
        t5 = lo.topk(5, -1).indices
        c1 += (t5[:, 0] == yb).sum().item(); c5 += (t5 == yb[:, None]).any(1).sum().item()
    N = len(X); loss = tot / N
    return dict(loss=loss, ppl=math.exp(loss), top1=c1 / N, top5=c5 / N)

def train(kind):
    set_seed()
    m = LM(kind).to(DEVICE)
    opt = torch.optim.Adam(m.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=1)
    hist = collections.defaultdict(list); best, bad, best_state = 1e9, 0, None
    t0 = time.time()
    for ep in range(1, EPOCHS + 1):
        m.train(); perm = torch.randperm(len(Xtr)); tl, k = 0.0, 0
        for i in range(0, len(perm), BATCH):
            idx = perm[i:i + BATCH]
            loss = F.cross_entropy(m(Xtr[idx]), ytr[idx])
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
            tl += loss.item() * len(idx); k += len(idx)
        va = evaluate(m, Xva, yva); sched.step(va["loss"])
        hist["train_loss"].append(tl / k); hist["val_loss"].append(va["loss"]); hist["val_top1"].append(va["top1"])
        print(f"[{kind}] ep{ep:02d} train {tl/k:.3f} val {va['loss']:.3f} ppl {va['ppl']:.1f} acc {va['top1']:.3f}", flush=True)
        if va["loss"] < best - 1e-4:
            best, bad = va["loss"], 0; best_state = {k_: v.clone() for k_, v in m.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE: break
    secs = time.time() - t0
    m.load_state_dict(best_state)
    return m, dict(hist), secs, ep

models, results, histories = {}, {}, {}
for kind in ["SimpleRNN", "BiRNN", "LSTM", "GRU"]:
    m, h, secs, ep = train(kind)
    te = evaluate(m, Xte, yte); tr = evaluate(m, Xtr, ytr); va = evaluate(m, Xva, yva)
    models[kind], histories[kind] = m, h
    results[kind] = dict(params=n_params(m), epochs_run=ep, best_epoch=int(np.argmin(h["val_loss"])) + 1,
                         train_time_s=secs, sec_per_epoch=secs / ep,
                         train_loss=tr["loss"], val_loss=va["loss"], test_loss=te["loss"],
                         test_ppl=te["ppl"], val_ppl=va["ppl"], train_ppl=tr["ppl"],
                         test_top1=te["top1"], test_top5=te["top5"])
json.dump(dict(dataset_stats=stats, results=results, histories=histories),
          open(os.path.join(OUT, "metrics.json"), "w"), indent=1, default=str)

# curves
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
for k, h in histories.items():
    ax[0].plot(h["val_loss"], label=k); ax[1].plot(h["train_loss"], label=k)
ax[0].set_title("Validation loss"); ax[1].set_title("Training loss")
for x in ax: x.set_xlabel("epoch"); x.legend()
plt.tight_layout(); plt.savefig(os.path.join(OUT, "loss_curves.png"), dpi=140); plt.close()

names = list(results)
fig, ax = plt.subplots(1, 4, figsize=(14, 3.4))
for x, (key, ttl) in zip(ax, [("test_ppl", "Test perplexity (lower better)"), ("test_top1", "Test top-1 accuracy"),
                              ("test_top5", "Test top-5 accuracy"), ("train_time_s", "Training time (s)")]):
    x.bar(names, [results[n_][key] for n_ in names], color=["#8c8c8c", "#c0504d", "#3b6ea5", "#4f9d69"])
    x.set_title(ttl, fontsize=10); x.tick_params(axis="x", rotation=30)
plt.tight_layout(); plt.savefig(os.path.join(OUT, "metric_comparison.png"), dpi=140); plt.close()
for k, r in results.items(): print(k, {a_: round(b_, 4) if isinstance(b_, float) else b_ for a_, b_ in r.items()})

# %% [markdown]
# ## 5. Text generation with temperature sampling

# %%
SEEDS = ["alice was beginning to get very tired of sitting",
         "the queen turned crimson with fury and shouted",
         "down the rabbit hole she fell and"]
TEMPS = [0.5, 1.0, 1.5]
GEN_LEN = 50

def tok(s): return re.findall(r"[a-z]+(?:'[a-z]+)*|[.,!?;:\"()-]", s.lower())

@torch.no_grad()
def generate(m, seed_text, temp, n=GEN_LEN, rng=None):
    m.eval()
    ctx = [stoi.get(t, 1) for t in tok(seed_text)]
    out = list(ctx)
    for _ in range(n):
        x = torch.tensor([([0] * SEQ_LEN + out)[-SEQ_LEN:]])
        lo = m(x)[0]
        lo[0] = -1e9; lo[1] = -1e9                  # never emit <pad>/<unk>
        p = F.softmax(lo / temp, -1)
        out.append(torch.multinomial(p, 1, generator=rng).item())
    return out[len(ctx):], ctx

def detok(ws):
    s = " ".join(itos[i] for i in ws)
    return re.sub(r" ([.,!?;:)])", r"\1", s).replace("( ", "(")

def gen_metrics(gen, ctx):
    full = ctx + gen
    d1 = len(set(gen)) / len(gen)
    bg = list(zip(gen[:-1], gen[1:])); d2 = len(set(bg)) / len(bg)
    tg = list(zip(full[:-2], full[1:-1], full[2:]))[len(ctx) - 1:]
    rep3 = 1 - len(set(tg)) / len(tg)                                # repeated trigrams
    valid_bg = np.mean([(a_, b_) in train_ngrams for a_, b_ in zip(full[len(ctx) - 1:-1], full[len(ctx):])])
    valid_tg = np.mean([t in train_trigrams for t in tg])
    maxrun = max(len(list(g)) for _, g in __import__("itertools").groupby(gen))
    return dict(distinct1=d1, distinct2=d2, rep_trigram=rep3, bigram_valid=float(valid_bg),
                trigram_valid=float(valid_tg), max_repeat_run=maxrun)

samples, gm = [], collections.defaultdict(list)
N_REP = 10    # metrics averaged over 10 stochastic runs per (model, seed, temp)
for kind, m in models.items():
    for si, sd in enumerate(SEEDS):
        for T in TEMPS:
            for r in range(N_REP):
                g = torch.Generator().manual_seed(1000 * si + r)
                gen, ctx = generate(m, sd, T, rng=g)
                gm[(kind, T)].append(gen_metrics(gen, ctx))
                if r == 0:
                    samples.append(dict(model=kind, seed=sd, temperature=T, text=sd + " | " + detok(gen)))

gen_summary = {}
for (kind, T), lst in gm.items():
    gen_summary[f"{kind}|{T}"] = {k: float(np.mean([d[k] for d in lst])) for k in lst[0]}
json.dump(dict(samples=samples, summary=gen_summary), open(os.path.join(OUT, "generation.json"), "w"), indent=1)

with open(os.path.join(OUT, "generated_samples.md"), "w", encoding="utf-8") as f:
    f.write("# Generated text samples (50 new words each; the text before `|` is the seed)\n\n")
    for sd in SEEDS:
        f.write(f"## Seed: *{sd}*\n\n")
        for T in TEMPS:
            f.write(f"### Temperature {T}\n\n")
            for kind in models:
                s = [x for x in samples if x["model"] == kind and x["seed"] == sd and x["temperature"] == T][0]
                f.write(f"- **{kind}**: {s['text']}\n")
            f.write("\n")

# temperature plots
fig, ax = plt.subplots(1, 4, figsize=(15, 3.6))
for x, (key, ttl) in zip(ax, [("distinct2", "Distinct-2 (diversity)"), ("rep_trigram", "Repeated-trigram rate"),
                              ("bigram_valid", "Bigrams seen in training text"), ("trigram_valid", "Trigrams seen in training text")]):
    for kind in models:
        x.plot(TEMPS, [gen_summary[f"{kind}|{T}"][key] for T in TEMPS], "o-", label=kind)
    x.set_title(ttl, fontsize=10); x.set_xlabel("temperature"); x.set_xticks(TEMPS)
ax[0].legend(fontsize=8); plt.tight_layout(); plt.savefig(os.path.join(OUT, "temperature_analysis.png"), dpi=140); plt.close()

print("\n=== generation metrics (mean over runs) ===")
for k, v in gen_summary.items(): print(k, {a_: round(b_, 3) for a_, b_ in v.items()})
print(open(os.path.join(OUT, "generated_samples.md"), encoding="utf-8").read()[:6000])
