# English → Urdu Machine Translation: Comparative Report

**Dataset:** [Parallel corpus for English & Urdu](https://www.kaggle.com/datasets/zainuddin123/parallel-corpus-for-english-urdu-language). After removing empty lines there are 24,524 sentence pairs. Most are short conversational sentences of 3–6 words.
**Code:** [`nmt_en_ur.ipynb`](nmt_en_ur.ipynb) (executed notebook) / [`nmt_en_ur.py`](nmt_en_ur.py) (same code as a script).
**Hardware:** Kaggle notebook on an NVIDIA Tesla T4. The full training log is in [`results/training_log.txt`](results/training_log.txt).

## 1. Setup

### Preprocessing
1. Loaded both text files into a Pandas DataFrame and dropped empty lines.
2. Tokenised the text:
   * English: lower-cased word tokens, with punctuation split off.
   * Urdu: Urdu punctuation (`۔ ؟ ،`) split off, then split on whitespace.
3. Split the data with a fixed seed: 80% train (19,619), 10% validation (2,452) and 10% test (2,453).
4. Built the vocabularies from the training split only. Both are capped at **6,000 tokens** and include `<pad> <sos> <eos> <unk>`. The English vocabulary has 5,135 entries and the Urdu one has 5,134.
5. Padded every source and target sequence to **16 tokens**. Target sequences are wrapped in `<sos> … <eos>`. This covers more than 99% of sentences without truncation.

The vocabulary and padding are shared by every model.

### Shared training configuration (identical for all models)

| Setting | Value |
|---|---|
| Embedding dim | 200 (matches GloVe 6B-200d) |
| Hidden size | 256 |
| Layers | 2 encoder + 2 decoder |
| Dropout | 0.3 |
| Optimiser | Adam, lr = 5e-4 |
| Batch size | 64 |
| Epochs | **50** |
| Loss | Cross-entropy, label smoothing 0.1, padding ignored |
| Gradient clipping | 1.0 |
| Decoding | Greedy, max 16 tokens |
| Checkpoint | Weights with the lowest validation loss |
| Seed | 42 |

### Models
* **RNN / LSTM:** Seq2seq encoder-decoder in the style of Cho et al. (2014).
  * The encoder is a 2-layer Elman RNN or LSTM. Sequences are packed so padding is ignored.
  * The final state initialises the decoder.
  * The top-layer context vector is also fed to the decoder at every step and to the output layer.
  * These models have no attention.
* **BiRNN / BiLSTM:** The same as above, but with a bidirectional encoder. The forward and backward final states of each layer are concatenated and projected with Linear + tanh to the decoder's hidden size.
* **Transformer:** `nn.Transformer` with 2 encoder and 2 decoder blocks, 8 attention heads, d_model 200 and a feed-forward size of 512. It uses sinusoidal positional encodings, padding masks and a causal decoder mask.
* **RNN + GloVe:** The same RNN, but the English embedding layer starts from `glove.6B.200d` vectors and is then fine-tuned. GloVe covers 98.2% of the English vocabulary (5,037 of 5,131 words).

## 2. Results on the held-out test set (2,453 sentences)

All scores are on a 0–100 scale. BLEU is corpus-level BLEU-4. ROUGE and METEOR are averaged over sentences.

| Model | **BLEU-4** | BLEU-1 | **ROUGE-1** | ROUGE-2 | **ROUGE-L** | **METEOR** | chrF | Best val loss | Params | Train time (50 ep) | s / epoch |
|---|---|---|---|---|---|---|---|---|---|---|---|
| RNN | 32.76 | 59.74 | 60.93 | 38.82 | 60.59 | 52.71 | 49.04 | 2.885 | 5.25 M | 162 s | 3.2 |
| BiRNN | 33.07 | 59.73 | 61.14 | 39.44 | 60.77 | 52.81 | 49.33 | 2.882 | 5.76 M | 177 s | 3.5 |
| LSTM | 37.78 | 64.40 | 64.92 | 43.80 | 64.42 | 57.52 | 54.04 | **2.747** | 6.94 M | 212 s | 4.2 |
| BiLSTM | 37.22 | 63.95 | 64.45 | 42.99 | 64.07 | 56.77 | 53.00 | 2.790 | 8.72 M | 246 s | 4.9 |
| **Transformer** | **39.70** | **65.08** | **65.75** | **45.54** | **65.12** | **58.80** | **56.99** | 2.824 | 4.88 M | 289 s | 5.8 |
| RNN + GloVe | 35.32 | 62.28 | 63.19 | 41.16 | 62.79 | 55.31 | 51.24 | 2.785 | 5.25 M | 161 s | 3.2 |

Notes on the metrics:
* **ROUGE** is computed in the notebook on Urdu word tokens. The common `rouge_score` package's default tokenizer drops non-Latin characters, so it would score every Urdu sentence as 0.
* **METEOR** uses exact and stem matches only, because there is no Urdu WordNet.
* **chrF** is a character-level check that does not depend on how words are tokenised.

Plots:
* [`results/loss_curves.png`](results/loss_curves.png): training and validation loss for all models.
* [`results/scores.png`](results/scores.png): test scores by model.
* [`results/glove_vs_random.png`](results/glove_vs_random.png): RNN with random vs. GloVe embeddings.

### Inference examples (a fixed set of test sentences; full set in [`results/sample_translations.csv`](results/sample_translations.csv))

| English | Reference | RNN | BiRNN | LSTM | BiLSTM | Transformer | RNN + GloVe |
|---|---|---|---|---|---|---|---|
| he is an italian | وہ ایک اطالوی ہے | وہ ایک excon ہے | وہ ایک انگریز ہے | وہ ایک ماہر ہے | وہ ایک ماہر ہے | **وہ ایک اطالوی ہے** | وہ ایک فرانسیسی ہے |
| zain is seldom home | زین شاذ و نادر ہی گھر ہوتا ہے۔ | زین گھر گھر ہے | زین گھر سے گھر ہے ۔ | زین گھر سے گھر ہے ۔ | زین گھر سے گھر ہے ۔ | **زین شاذ و نادر ہی گھر ہے ۔** | زین گھر سے گھر ہے ۔ |
| i need a new bicycle | مجھے ایک نئی سائیکل چاہیے | **مجھے ایک نئی سائیکل چاہیے** | **مجھے ایک نئی سائیکل چاہیے** | میں ایک نئی سائیکل کی ضرورت ہے | میں ایک نئی الماری کی ضرورت ہے | **مجھے ایک نئی سائیکل چاہیے** | **مجھے ایک نئی سائیکل چاہیے** |
| i can sleep anywhere | میں کہیں بھی سو سکتا ہوں | میں کہیں بھی جا سکتا ہوں ۔ | میں کہیں کہیں ہوں | میں بھی سو سکتا ہوں | میں کہیں بھی جا سکتا ہوں | **میں کہیں بھی سو سکتا ہوں** | **میں کہیں بھی سو سکتا ہوں** |
| were leaving | جا رہے تھے | مورخ تھے | **جا رہے تھے** | **جا رہے تھے** | **جا رہے تھے** | **جا رہے تھے** | چھپے تھے |

Custom sentences that are not in the dataset ([`results/custom_translations.csv`](results/custom_translations.csv)):

| English | RNN | LSTM | BiLSTM | Transformer | RNN + GloVe |
|---|---|---|---|---|---|
| i am happy | میں خوش ہوں | میں خوش ہوں | میں خوش ہوں | میں خوش ہوں | میں خوش ہوں |
| where is my book | میری کتاب کہاں ہے ؟ | میری کتاب کہاں ہے ؟ | میری کتاب کہاں ہے ؟ | میری کتاب کہاں ہے | میری کتاب کہاں ہے ؟ |
| what is your name | ہائے ✗ | ہائے ✗ | ہائے ✗ | **آپ کا نام کیا ہے** | ہائے ✗ |
| we are going home | ہم گھر جا رہے ہیں | ہم گھر جا رہے ہیں | ہم گھر جا رہے ہیں | ہم گھر جا رہے ہیں | ہم گھر جا رہے ہیں |

All recurrent models translate "what is your name" as "ہائے" ("hi"). They have learned a greeting-like mapping from the fixed context vector. The Transformer attends to the individual words and gets the translation right.

## 3. Part 4: random vs. pre-trained GloVe embeddings (RNN)

| | RNN (random emb.) | RNN + GloVe | Δ |
|---|---|---|---|
| BLEU-4 | 32.76 | **35.32** | **+2.56** |
| ROUGE-L | 60.59 | **62.79** | +2.20 |
| METEOR | 52.71 | **55.31** | +2.60 |
| chrF | 49.04 | **51.24** | +2.20 |
| Best validation loss | 2.885 (epoch 48) | **2.785** (epoch 48) | −0.10 |
| Epochs to reach the random model's best val loss (2.885) | 48 | **22** | **2.2× faster** |
| Wall-clock training time (50 epochs) | 162 s | 161 s | ≈ same |
| Time per epoch | 3.23 s | 3.22 s | ≈ same |

### Discussion
* **Performance.** GloVe initialisation improves every metric by 2–2.6 points (BLEU +2.6, METEOR +2.6). That is almost 60% of the gap between the plain RNN and the LSTM, with no extra parameters. The corpus is small (about 20k sentences), so many English words appear only a few times. With random initialisation the model must learn their meaning from those few examples. GloVe vectors are trained on 6B tokens and already place related words close together (*italian / french / english*, *happy / glad*). The encoder therefore gets meaningful input from the first epoch and generalises better to rare words.
* **Training time.** The cost per epoch is identical (3.2 s) because the architecture and parameter count do not change. Reading the GloVe file and building the embedding matrix is a one-off cost of a few seconds. Convergence is much faster: the GloVe model reaches the random model's *best* validation loss at epoch 22 instead of epoch 48. It is also ahead from the first few epochs (epoch 10 val loss: 3.09 vs 3.19). To reach a given quality you need roughly half the training time.
* **Limits.** The gains come only from the encoder, because GloVe does not cover Urdu. They are bounded by the RNN's fixed-size context vector. Some outputs show GloVe's semantic neighbours being confused, e.g. *italian* → *فرانسیسی* (French). Similar words have similar vectors, and the small decoder cannot always tell them apart.

## 4. Conclusion: which model performed best and why?

**The Transformer is the best model.** It has the highest score on every metric: BLEU 39.7, ROUGE-L 65.1, METEOR 58.8 and chrF 57.0. It also has the *fewest* parameters (4.9 M). The ranking is

**Transformer > LSTM ≈ BiLSTM > RNN + GloVe > BiRNN ≈ RNN**

Why:
1. **Attention replaces the bottleneck.** Every recurrent model squeezes the source sentence into one 256-d vector. The Transformer's multi-head cross-attention lets each Urdu output word look directly at the relevant English words. This shows on rare and content words that have to be copied over precisely, e.g. *italian → اطالوی*, *seldom → شاذ و نادر*, *your name → آپ کا نام*, which the recurrent models replace with frequent words. It also shows in ROUGE-2 and chrF, which reward correct word order and correct word forms; there the Transformer leads by 1.7 and 3 points.
2. **Gating matters for RNNs.** LSTMs beat vanilla RNNs by about 5 BLEU. The gates protect the gradient and the memory, so the final encoder state keeps more of the sentence. The vanilla RNN's state is dominated by the last few words.
3. **Bidirectionality helps little here.** BiRNN gains 0.3 BLEU over RNN, and BiLSTM is slightly *worse* than LSTM (−0.6). Without attention, the decoder only sees the final states, and the sentences are short (about 5 tokens), so the forward state already sees most of the sentence. BiLSTM also has 26% more parameters. It overfits sooner: its best validation loss came at epoch 29 and then rose, while the training loss was the lowest of all models.
4. **The Transformer had not finished learning.** Its validation loss was still falling at epoch 50, and its best checkpoint is the last one. The other models had plateaued. With more epochs or a learning-rate warm-up schedule, the gap would probably grow. Its higher time per epoch (5.8 s vs 3–5 s) comes from the self-attention and cross-attention layers, which do more work per token than a 2-layer RNN on such short sequences. It is still under 5 minutes in total on a T4.

Overall, for this small, short-sentence English→Urdu corpus, attention matters more than recurrence type, and recurrence type matters more than bidirectionality. Pre-trained embeddings are a cheap way to improve a weak model when data is scarce.

## 5. Reproducing

* **Kaggle (GPU):** push the kernel in [`kaggle/`](kaggle). This trains everything from scratch, about 25 minutes on a T4.
  ```bash
  kaggle kernels push -p kaggle --accelerator NvidiaTeslaT4
  ```
* **Locally:** put `english-corpus.txt`, `urdu-corpus.txt` and `glove.6B.200d.txt` under `DATA_DIR`.
  ```bash
  DATA_DIR=data python nmt_en_ur.py
  ```
  To evaluate the trained weights in `results/` without retraining, set `PRETRAINED_DIR=results` (this is how `nmt_en_ur.ipynb` was executed).
