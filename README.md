# NLP Assignment 3: English → Urdu Neural Machine Translation

This repo compares RNN, BiRNN, LSTM, BiLSTM and Transformer encoder-decoder models, and an RNN trained with random vs. GloVe English embeddings. All models were trained for 50 epochs with identical settings on a Kaggle T4 GPU.

| File | What it is |
|---|---|
| [`REPORT.md`](REPORT.md) | **Report**: comparison table, inference examples, GloVe analysis, conclusion |
| [`nmt_en_ur.ipynb`](nmt_en_ur.ipynb) | Executed notebook: preprocessing, models, training loop, metrics, visualisation |
| [`nmt_en_ur.py`](nmt_en_ur.py) | Same code as a plain Python script (jupytext percent format) |
| [`kaggle/`](kaggle) | Kaggle kernel (notebook + metadata) used for GPU training |
| [`results/`](results) | Scores (`results.csv`, `results.json`), sample translations, plots, training log, trained weights (`*.pt`) |

## Test-set results

| Model | BLEU-4 | ROUGE-L | METEOR | chrF |
|---|---|---|---|---|
| RNN | 32.76 | 60.59 | 52.71 | 49.04 |
| BiRNN | 33.07 | 60.77 | 52.81 | 49.33 |
| LSTM | 37.78 | 64.42 | 57.52 | 54.04 |
| BiLSTM | 37.22 | 64.07 | 56.77 | 53.00 |
| **Transformer** | **39.70** | **65.12** | **58.80** | **56.99** |
| RNN + GloVe | 35.32 | 62.79 | 55.31 | 51.24 |
