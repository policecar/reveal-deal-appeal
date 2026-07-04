# Win/No-Win Prediction from Discovery-Call Transcripts — Findings

**Date:** 2026-07-04 · **Branch:** `claude/code-improvement-discussion-4qsm1o` · **Metrics:** `checkpoints/cv_metrics.json`

## TL;DR

Five modeling approaches across three representation families all score at
chance level (pooled AUROC 0.46–0.57) on predicting deal outcome from
discovery-call transcripts. The dataset yields **8 usable win examples vs
196 no-wins**, and the popular escape hatches were tested and eliminated:
class-imbalance handling didn't help, and covering the *whole* call instead
of the first 3 minutes didn't help either. The recommendation is to stop
investing in this target/representation combination and re-aim the (now
solid) pipeline at a nearer-term, more frequent label — e.g. intro-call →
discovery-call conversion — or at LLM-extracted structured features.

## The data

Source: `data/transcripts.xlsx`, sheet `discovery calls`, 259 labeled rows.

| Label | raw | after filtering |
|---|---|---|
| No | 242 | 196 |
| Win | 10 | 3 |
| Potential Win | 5 | 3 |
| Win - MVP never started | 2 | 2 |

Filtering (both steps verified by content inspection):

- **40 rows have no usable transcript** (empty cell) — including 7 win-ish
  rows. Unrecoverable without re-transcription; nothing to classify.
- **17 rows are stubs under 750 words** (~5 min of speech): fragments where
  the recording died after a minute, and failed transcriptions (a 12-minute
  call with 12 words of ASR garbage). The two win-ish rows among them are
  pure logistics ("can we switch to Zoom") with zero deal content. The
  `Length` column cannot catch these — a 20-minute call can carry 601
  garbled words — hence the word-count floor.

Everything ≠ "No" currently counts as a win; whether *Potential Win*
belongs there is an open business question (it is 3 of the 8 positives).

**Call length vs model input** (ModernBERT tokenizer, measured): median call
= 10,421 tokens, max 17,345. At the working `max_length` of 1024 a model
sees **9.6 % of the corpus — the first ~3 minutes of each call**. Even at
ModernBERT's 8192 ceiling only 20.6 % of calls fit entirely. Every
experiment before this session (mpnet @ 384 → 3.6 %) had judged calls by
their opening small talk.

## Pipeline fixes made along the way

Results from before these fixes should be considered invalid.

| Fix | Commit |
|---|---|
| Config pointed at a nonexistent sheet/column (`Sales calls` / `Win /No-Win`) | `c9e37d1` |
| presidio deps missing from requirements; numpy pin uninstallable next to numba | `c6de9be` |
| **Presidio silently ignored ORGANIZATION entities** (default `labels_to_ignore`) — org names leaked through "anonymized" data; now redacted | `9f943cf` |
| SetFit head crashed on construction (LazyLinear xavier-init); explicit `in_features` | `6efe3f0` |
| 2048-token training OOMs 24 GB MPS; reduced sizes silently swap-thrashed at ~280 s/step; gradient checkpointing → 7.7 s/step @ ~8 GB | `3faf024` |
| z-score outlier filter → minimum-length floor (750 words) | `506806e` |
| Imbalance handling: pair oversampling + class-weighted CE, focal-loss option (`model.focal_gamma`) | `30ff0ab` |
| CV harness: TF-IDF & frozen-embedding baselines, whole-call chunked baselines, AUROC + win-rank reporting | `bd8f39f`, `7727c2a` |

Plus 11 unit tests (label mapping, index tracking, head construction,
losses, chunk pooling) and earlier handoff fixes (ClassLabel ordering,
original-index tracking, inverted report names, config-driven ModernBERT).

Anonymization caveat: NER-based redaction is best-effort — "humanitec"
itself still appears in ~80 % of rows (lowercased conversational text);
a deterministic deny-list recognizer would close that if strictness matters.

## Experiments and results

All results: 5-fold stratified CV over the pooled 204 calls, fresh model
per fold, identical folds across models, seed 1537. Hardware: Apple
Silicon, 24 GB (SetFit CV ≈ 80 min; all baselines ≈ 20 min, no GPU needed).

| # | Model | Input seen | macro-F1 | F2 (Win) | pooled AUROC |
|---|---|---|---|---|---|
| 1 | SetFit + ModernBERT, undersampling, unweighted CE | first 1024 tokens | 0.489 ± 0.003 | 0.000 | — |
| 2 | SetFit + ModernBERT, oversampling, class-weighted CE | first 1024 tokens | 0.487 ± 0.006 | 0.000 | — |
| 3 | TF-IDF (1–2 grams) + LR `balanced` | whole call (lexical) | 0.490 ± 0.003 | 0.000 | 0.459 |
| 4 | Frozen ModernBERT embedding + LR `balanced` | first 1024 tokens | 0.524 ± 0.107 | 0.100 ± 0.200 | 0.557 |
| 5 | Chunked frozen embeddings, mean-pooled + LR | **whole call** | 0.479 ± 0.067 | 0.077 ± 0.154 | 0.555 |
| 6 | Chunked frozen embeddings, max chunk probability | **whole call** | 0.262 ± 0.008 | 0.214 ± 0.049 | 0.571 |

Reading guide: AUROC 0.5 = coin flip. The most tangible view is **win
ranks** — where the 8 real wins land when all 204 calls are sorted by
predicted win probability (rank 1 = most win-like):

```
TF-IDF:        12,  78,  94, 118, 123, 133, 153, 174
frozen (3min): 13,  31,  66,  98, 101, 110, 122, 189
chunked mean:  18,  60,  67,  77, 104, 113, 137, 157
chunked max:   31,  44,  60,  72,  90, 125, 137, 149
```

That is approximately what uniform random scattering looks like.

Notes:
- Run 1 collapsed to predicting No-Win for every example. Run 2's imbalance
  fixes moved it off pure collapse (occasional Win predictions) but caught
  0 of 8 wins — the embedding space has no win-separating direction for the
  head to exploit.
- Model 6's nonzero F2 is a threshold artifact (max over ~11 chunk
  probabilities inflates scores past 0.5, flooding false positives — see
  its macro-F1); its *ranking* is still chance-level.
- **The truncation hypothesis is rejected**: whole-call coverage (5, 6)
  scores the same as first-3-minutes coverage (4) with the same encoder.
- **Finetuning adds nothing**: SetFit (2) does not beat its own frozen
  starting point (4).

## Where that leaves us

**The negative result is robust.** Lexical features, truncated embeddings,
whole-call embeddings, chunk-max scoring, and contrastive finetuning agree.
With 8 positives, deal outcome is not extractable from these transcripts —
either the signal isn't in the call (wins hinge on pricing, timing,
competitors, committee decisions that happen elsewhere), or 8 examples are
too few to find it. Caveats in fairness: with 8 positives even AUROC
carries roughly ±0.1 noise, the "win" class mixes three label semantics,
and anonymization (lowercasing, entity stripping) may erase weak signals.

**Spending GPU time on this configuration is not justified.** Longer
context, bigger models, or hyperparameter search all polish a
representation whose ceiling is currently indistinguishable from chance.

### Recommended next moves, ranked

1. **Re-target the pipeline: funnel progression instead of final outcome.**
   The sheet has `1st Call` / `Discovery Call` columns; "did this intro
   call convert to a discovery call?" is a nearer-term outcome that each
   call plausibly causes, with (likely) far better class balance. The
   entire harness — anonymization, CV, baselines, ranking metrics — reuses
   with a different label column. An afternoon of work.
2. **LLM-extracted structured features → logistic regression.** At 204
   examples, features like "explicit next step agreed", "budget discussed",
   "champion present", "objection count" routinely beat dense embeddings.
   Transcripts are already anonymized, lowering the barrier to API-model
   extraction.
3. **Recover the 7 win-ish rows with missing/broken transcripts** (data
   ops: find the recordings, re-transcribe). Doubles the positive class for
   any future attempt — necessary but probably not sufficient given
   chance-level AUROC.
4. **Decide the label taxonomy** — whether *Potential Win* is a positive
   (currently 3 of 8), business call.

### Open hygiene items (unchanged from handoff)

`breakpoint()` at the end of refine.py; unused `training:` config block;
Makefile references a missing requirements-dev.txt; 21 Dependabot alerts
(1 high) on the default branch; `_create_labels` accepts any unexpected
string as a win (normalize + raise).

## Reproduction

```bash
git checkout claude/code-improvement-discussion-4qsm1o
uv venv --python 3.11 && uv pip install -r requirements.txt
uv pip install "en_core_web_lg @ https://github.com/explosion/spacy-models/releases/download/en_core_web_lg-3.8.0/en_core_web_lg-3.8.0-py3-none-any.whl"
python tmp/preprocess.py                 # xlsx -> data/mauzo (Presidio, ~3 min)
python src/crossval.py --baselines-only  # models 3-6, ~20 min CPU/MPS
python src/crossval.py                   # + SetFit CV, ~80 min on 24GB MPS
```

Config knobs in `src/config.yaml` (`model:`): `max_length`, `batch_size`,
`gradient_checkpointing`, `focal_gamma`. Metrics land in
`checkpoints/cv_metrics.json`.
