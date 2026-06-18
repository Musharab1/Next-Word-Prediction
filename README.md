# 🔤 Next Word Predictor

A deep learning-based next-word prediction system built **entirely from scratch using pure NumPy** — no TensorFlow, no PyTorch. Trained on a custom text corpus and served through an interactive Gradio UI with real-time suggestions and online learning.

---

## ✨ Features

- **Pure NumPy neural network** — forward pass, backpropagation, and SGD all hand-coded
- **3-layer architecture** — Embed → Dense(256, Leaky ReLU) → Dense(128, Leaky ReLU) → Softmax
- **N-gram sliding window** — uses the last N=3 words as context to predict the next word
- **Xavier/He weight initialization** — for stable training from the start
- **Top-K inference** — returns the top 5 most probable next words with confidence scores
- **Online learning** — model updates in real time from new user-typed sentences
- **Persistent storage** — embeddings, weights, and vocab saved to Google Drive
- **Gradio UI** — live word suggestion buttons that update on every keystroke

---

## 🏗️ Architecture

```
Input (context window, N=3 words)
        │
        ▼
Embedding Lookup  →  concat  →  shape: (384,)   [3 × 128-dim embeddings]
        │
        ▼
Dense(256)  +  Leaky ReLU
        │
        ▼
Dense(128)  +  Leaky ReLU
        │
        ▼
Dense(vocab_size)  +  Softmax
        │
        ▼
Top-K Predictions
```

---

## 📁 Project Structure

```
Next_word_predictor/
├── sentences.txt           # Training corpus (one sentence per line, UTF-8)
├── vocab.pkl               # Pickled word → ID dictionary
├── embeddings.npy          # Trained embedding table (vocab_size × 128)
├── model_parameters.pkl    # Trained weights W1,b1,W2,b2,W3,b3
└── Deep_learning_project.ipynb  # Main Colab notebook
```

---

## 🚀 Getting Started

### 1. Open in Google Colab

Upload or open `Deep_learning_project.ipynb` in [Google Colab](https://colab.research.google.com/).

### 2. Mount Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

### 3. Set file paths

Edit the config block at the top of the notebook to point to your Drive folder:

```python
DATASET_FILE      = '/content/drive/MyDrive/Next_word_predictor/sentences.txt'
EMBEDDINGS_FILE   = '/content/drive/MyDrive/Next_word_predictor/embeddings.npy'
VOCAB_FILE        = '/content/drive/MyDrive/Next_word_predictor/vocab.pkl'
PARAMETERS_FILE   = '/content/drive/MyDrive/Next_word_predictor/model_parameters.pkl'
EMBED_DIM = 128
N = 3  # context window size
```

### 4. Run all cells

The notebook will:
- **Load** saved model files from Drive if they exist
- **Train** a new model (300,000 iterations) if no saved files are found
- **Launch** the Gradio interface automatically

---

## 🧠 How It Works

### Preprocessing
- Text is lowercased and stripped of all non-alphabetic characters
- Words appearing fewer than **3 times** are excluded from the vocabulary and replaced with `<UNK>`

### Training
- Sliding N-gram windows create `(context, target)` pairs from the corpus
- Mini-batches of size 64 are sampled randomly at each iteration
- SGD with learning rate `0.01` updates weights and the embedding table simultaneously
- Loss is printed every 5,000 iterations

### Inference
- The last N words of the input are looked up in the embedding table
- Their vectors are concatenated and fed through the network
- Softmax output is sorted; the top-K valid words (excluding `<UNK>`, empty strings, low-confidence predictions) are returned

### Online Learning
- When the user clicks **"Learn from this sentence"**, the model runs 20 gradient update steps on that sentence at `lr=0.005`
- Updated parameters are immediately saved back to Google Drive

---

## 🖥️ User Interface

| Element | Description |
|---|---|
| **Text box** | Type your sentence here |
| **5 suggestion buttons** | Click to append a predicted word (shows word + confidence %) |
| **Learn button** | Trains the model on your current sentence |
| **Status box** | Confirms what was learned |

---

## ⚙️ Hyperparameters

| Parameter | Value |
|---|---|
| Embedding dimension | 128 |
| Context window (N) | 3 |
| Hidden layer 1 | 256 neurons |
| Hidden layer 2 | 128 neurons |
| Activation | Leaky ReLU (α=0.01) |
| Loss | Cross-entropy |
| Optimizer | SGD |
| Learning rate | 0.01 (training), 0.005 (online) |
| Batch size | 64 |
| Training iterations | 300,000 |
| Min word frequency | 3 |

---

## 📦 Dependencies

```bash
pip install gradio numpy
```

> All other code uses Python standard library (`re`, `collections`, `pickle`).

---

## 📌 Notes

- The model trains from scratch if no saved files are found on Drive — this takes several minutes on Colab CPU.
- For best predictions, use a large and diverse `sentences.txt` corpus.
- The Gradio share link expires after 1 week; re-run the last cell to get a fresh link.

---



**2022-EE-251 & 2022-EE-274**  
Department of Electrical Engineering  
University of Engineering and Technology, Lahore  
Course: Deep Learning — CEP Project  
Supervisor: Dr. Umar Rashid
