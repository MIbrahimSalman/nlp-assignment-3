# Project context

**Student:** Mohammad Ibrahim Salman · **Roll Number:** 23L-0723 · **Course:** Natural Language Processing

Background for anyone (person or AI assistant) picking up this repo: what the assignment asks for, what has been
done, how the code is organised, the design decisions and why, and how to re-run things.

## Assignment (NLP Assignment 3)

English → Urdu machine translation on the Kaggle dataset
[`zainuddin123/parallel-corpus-for-english-urdu-language`](https://www.kaggle.com/datasets/zainuddin123/parallel-corpus-for-english-urdu-language).

1. **Preprocessing:** load the data with Pandas, tokenise it, and pad input and output to a fixed length.
2. **Models:** encoder-decoder translators: RNN, BiRNN, LSTM, BiLSTM and Transformer (multi-head attention).
   * Every model uses the same core training configuration.
   * Every model trains for at least 50 epochs.
   * Vocabulary size and padding are the same for every model.
3. **Evaluation:** translate a fixed set of test sentences with every model.
   * Score them with BLEU plus at least one more metric (ROUGE or METEOR).
   * Compare the scores and show inference examples.
4. **Pre-trained embeddings:** train the RNN twice, once with random English embeddings and once with GloVe.
   * Compare the scores and the training time.
5. **Deliverables:** a notebook or script, plus a report with a comparison table and a conclusion about which model is best and why.

**Status: complete.** All parts are done, trained and documented.

## Repository layout

| Path | Purpose |
|---|---|
| `nmt_en_ur.py` | **Source of truth** for the code (jupytext "percent" format: `# %%` cells, `# %% [markdown]` text cells) |
| `23L-0723_NLP_Assignment-3.ipynb` | Executed notebook generated from the `.py` (outputs are from evaluating the Kaggle-trained weights) |
| `kaggle/nmt_en_ur.ipynb` | Un-executed notebook that was pushed to Kaggle for GPU training |
| `kaggle/kernel-metadata.json` | Kaggle kernel config: GPU on, internet on, dataset sources (corpus + GloVe) |
| `23L-0723_NLP_Assignment-3.md` / `.pdf` | Final report (Markdown and LaTeX-compiled PDF; LaTeX source in `report/`): setup, results table, examples, GloVe discussion, conclusion |
| `README.md` | Short overview + headline results table |
| `results/results.csv` | One row per model: all metrics, best val loss/epoch, params, train time |
| `results/results.json` | Config, scores, per-epoch train/val loss and epoch times for each model, Part 4 stats |
| `results/sample_translations.csv` | 15 fixed test sentences translated by every model |
| `results/custom_translations.csv` | 5 hand-written sentences translated by every model |
| `results/*.png` | Loss curves, score bar chart, RNN random vs GloVe curves |
| `results/training_log.txt` | stdout of the full Kaggle GPU training run |
| `report/` | LaTeX source of the PDF report (`23L-0723_NLP_Assignment-3.tex`, compile with XeLaTeX for Urdu) and `make_figures.py`, which draws its figures from `results/results.json` |
| `results/*.pt` | Trained weights (best-val-loss checkpoints) for all six models |

**Editing workflow:** edit `nmt_en_ur.py`, then regenerate the notebooks:

```bash
jupytext --to ipynb nmt_en_ur.py -o kaggle/nmt_en_ur.ipynb
```

To refresh the executed notebook without retraining, run that file with `PRETRAINED_DIR=results` (see "Running").

## Data and preprocessing

* The dataset is two aligned files, `Dataset/english-corpus.txt` and `Dataset/urdu-corpus.txt`.
  * After dropping empty lines there are 24,524 pairs.
  * They are short conversational sentences (median about 5 tokens).
* **Tokenisers:**
  * English (`tok_en`): lower-cased, regex word tokens, punctuation split off.
  * Urdu (`tok_ur`): `۔ ؟ ، ! ? . , "` split off, then split on whitespace.
* **Split:** 80/10/10 with seed 42, giving 19,619 train, 2,452 val and 2,453 test pairs. Vocabularies are built from train only.
* **Vocabulary:** a custom `Vocab` class with the cap `MAX_VOCAB = 6000` for both languages.
  * Actual sizes: English 5,135, Urdu 5,134.
  * Special tokens: `<pad>=0, <sos>=1, <eos>=2, <unk>=3`.
* **Padding:** `MAX_LEN = 16` for source and target. The target is `<sos> … <eos>`, so it holds at most 14 words. This covers more than 99% of sentences.

## Shared configuration (`CFG` in the code)

| Setting | Value |
|---|---|
| Embedding dim | 200 (chosen to match GloVe 6B-200d) |
| Hidden size | 256 |
| Layers | 2 encoder + 2 decoder |
| Transformer | 8 heads, feed-forward size 512 |
| Dropout | 0.3 |
| Optimiser | Adam, lr 5e-4 |
| Batch size | 64 |
| Epochs | 50 |
| Gradient clipping | 1.0 |
| Loss | Cross-entropy, label smoothing 0.1, padding ignored |
| Decoding | Greedy |
| Checkpoint | Lowest validation loss |
| Seed | 42 |

## Models

* **`Seq2SeqRNN(cell, bidirectional, pretrained)`** covers RNN, BiRNN, LSTM, BiLSTM and RNN + GloVe.
  * The encoder uses packed sequences, so padding does not affect the final state.
  * For bidirectional encoders, the forward and backward states of each layer are concatenated, then passed through Linear + tanh down to 256.
  * The decoder is in the style of Cho et al. (2014): the context vector (top encoder layer) is concatenated to the decoder input at every step and to the decoder output before the vocabulary projection.
  * **No attention.** This is deliberate: it gives the classic seq2seq baseline, so the gain from attention is visible in the comparison.
* **`Seq2SeqTransformer`** wraps `nn.Transformer` (batch_first).
  * Sinusoidal positional encodings; embeddings are scaled by sqrt(d).
  * Padding masks and a causal decoder mask.
* **GloVe:** `glove.6B.200d.txt` from the Kaggle dataset `rtatman/glove-global-vectors-for-word-representation`.
  * It covers 98.2% of the English vocabulary (5,037 of 5,131 words).
  * Words not in GloVe get N(0, 0.1) vectors.
  * The embeddings are fine-tuned during training, not frozen.

## Metrics

* **BLEU-4:** NLTK `corpus_bleu` with smoothing method1. BLEU-1 is also reported.
* **ROUGE-1/2/L:** implemented by hand on word tokens. The `rouge_score` package's default tokenizer strips non-Latin text, so it would give Urdu a score of 0.
* **METEOR:** NLTK, called with a dummy `_NoWordNet`.
  * There is no Urdu WordNet, so only exact and stem matches count.
  * This also means no `nltk.download` is needed.
* **chrF:** implemented by hand (character 1–6-grams, β = 2). It is a check that does not depend on tokenisation.

## Results (test set, see `23L-0723_NLP_Assignment-3.md` for the full table and discussion)

| Model | BLEU-4 | ROUGE-L | METEOR | chrF | Train time |
|---|---|---|---|---|---|
| RNN | 32.76 | 60.59 | 52.71 | 49.04 | 162 s |
| BiRNN | 33.07 | 60.77 | 52.81 | 49.33 | 177 s |
| LSTM | 37.78 | 64.42 | 57.52 | 54.04 | 212 s |
| BiLSTM | 37.22 | 64.07 | 56.77 | 53.00 | 246 s |
| **Transformer** | **39.70** | **65.12** | **58.80** | **56.99** | 289 s |
| RNN + GloVe | 35.32 | 62.79 | 55.31 | 51.24 | 161 s |

Key findings:
* The **Transformer is best on all metrics** and has the fewest parameters (4.9 M). Attention removes the fixed-vector bottleneck.
  * Its validation loss was still falling at epoch 50, so it is undertrained relative to the others.
* **LSTM beats RNN by about 5 BLEU.** Bidirectional encoders barely help without attention.
  * BiLSTM overfits after epoch 29.
* **GloVe improves the RNN by 2–2.6 points** on every metric, at the same cost per epoch.
  * It reaches the random-embedding RNN's best validation loss at epoch 22 instead of 48.

## Running

**On Kaggle (full training, about 25 minutes on a T4).** The `KAGGLE_API_TOKEN` environment variable is enough for the `kaggle` CLI. The kernel id is `mibrahimsalman022/nlp-assignment-3-english-urdu-nmt`.

```bash
jupytext --to ipynb nmt_en_ur.py -o kaggle/nmt_en_ur.ipynb
```

```bash
kaggle kernels push -p kaggle --accelerator NvidiaTeslaT4
```

```bash
kaggle kernels status mibrahimsalman022/nlp-assignment-3-english-urdu-nmt
```

```bash
kaggle kernels output mibrahimsalman022/nlp-assignment-3-english-urdu-nmt -p results
```

The Kaggle API returns the output files and a JSON log, but not the executed notebook. The log was converted to
`results/training_log.txt` by keeping only its stdout entries.

**Locally.** The code finds data files by searching `DATA_DIR`, `/kaggle/input`, `data/` and `.` recursively.

```bash
DATA_DIR=path/to/data python nmt_en_ur.py
```

Environment variables:
* `EPOCHS`: override the epoch count, e.g. `EPOCHS=1` for a smoke test. A 1-epoch run of all models takes a few minutes on CPU.
* `ONLY_MODELS`: comma-separated model names to run, e.g. `RNN,Transformer`.
* `OUT_DIR`: where outputs go. The default is `/kaggle/working` on Kaggle and `outputs/` elsewhere.
* `PRETRAINED_DIR=results`: skip training, load the `.pt` weights and histories, and run evaluation and plots. This is how the committed `23L-0723_NLP_Assignment-3.ipynb` was produced:

```bash
PRETRAINED_DIR=results DATA_DIR=path/to/data jupyter nbconvert --to notebook --execute kaggle/nmt_en_ur.ipynb --output-dir . --output 23L-0723_NLP_Assignment-3.ipynb
```

Running locally reproduced the Kaggle scores and sample translations exactly.

## Possible next steps (not required by the assignment)

* Add Bahdanau or Luong attention to the recurrent models, to compare attention vs. no attention within one architecture family.
* Train the Transformer longer or with a warm-up learning-rate schedule. It had not converged at 50 epochs.
* Use beam search instead of greedy decoding.
* Use subword tokenisation (BPE/SentencePiece) to reduce `<unk>` and handle Urdu morphology.

## Rebuilding the PDF report

Needs XeLaTeX with polyglossia/bidi, plus the Noto Serif, Noto Sans, Noto Sans Mono and Noto Naskh Arabic fonts.
On Debian/Ubuntu these come from `texlive-xetex`, `texlive-latex-extra`, `texlive-pictures`, `texlive-lang-arabic` and `fonts-noto-core`.

```bash
cd report && python3 make_figures.py && xelatex 23L-0723_NLP_Assignment-3.tex && xelatex 23L-0723_NLP_Assignment-3.tex && xelatex 23L-0723_NLP_Assignment-3.tex && cp 23L-0723_NLP_Assignment-3.pdf ..
```
