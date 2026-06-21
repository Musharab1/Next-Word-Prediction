# Next-Word Predictor

A multi-layer neural network, built from scratch using NumPy, that predicts the next word in a sentence based on the preceding `N` words. The model is trained on a custom text corpus and served through an interactive Gradio web interface that also supports online (live) learning from user input.

**Submitted to:** Dr. Umar Rashid

**Submitted by:**
- Musharab Sabeen (2022-EE-251)
- Ahsan bin Asif (2022-EE-274)

## Objective

Train a multi-layer neural network to approximate the next-word probability distribution for a sequence of text, given the preceding `N` words as context.

## Project Workflow

1. **Data Loading** – Sentences are read from a text file (`sentences.txt`) stored on Google Drive at:
   ```
   /content/drive/MyDrive/Next_word_predictor/sentences.txt
   ```
2. **Tokenization & Preprocessing** – Text is lowercased, stripped of non-alphabetic characters, and split into word tokens.
3. **Vocabulary Building** – A vocabulary is constructed from all tokens that occur **at least 2 times**. Words below this frequency (and unseen words at inference time) are mapped to a special `<UNK>` token.
4. **Word Embeddings** – Each vocabulary word is assigned a trainable embedding vector (initialized randomly with Xavier-style scaling).
5. **Training Example Generation** – For every position in the token stream, the preceding `N` words form the context, and the following word is the target label.
6. **Model Training** – A fully custom feed-forward neural network (implemented with raw NumPy, no deep learning frameworks) is trained using mini-batch gradient descent with backpropagation, which also updates the word embeddings.
7. **Model Persistence** – Trained embeddings, network parameters, and the vocabulary are saved to Google Drive (and reloaded automatically on the next run if available, skipping retraining).
8. **Interactive UI** – A Gradio app allows users to type text, view live next-word suggestions with confidence scores, click suggestions to auto-complete, and trigger online learning on the typed sentence.

## Neural Network Architecture

```
Input:    Concatenation of N word embeddings  → shape (N * embed_dim,)
Hidden 1: Dense(256) + Leaky ReLU
Hidden 2: Dense(128) + Leaky ReLU
Output:   Dense(vocab_size) + Softmax
Loss:     Cross-Entropy
```

With the default configuration (`N = 3`, `EMBED_DIM = 128`), the input layer has 384 units.

### Core Components (implemented from scratch)

- Leaky ReLU activation and its derivative
- Softmax activation
- Parameter initialization (He/Xavier-style scaling)
- Forward propagation through 2 hidden layers
- Cross-entropy cost computation
- Backward propagation (including gradients flowing back into the embedding table)
- Parameter update step (gradient descent)
- Embedding table update step
- Mini-batch training loop (`nn_model`)
- Top-k prediction function

## Training Configuration

| Hyperparameter        | Value     |
|------------------------|-----------|
| Context window (N)     | 3         |
| Embedding dimension    | 128       |
| Hidden layer 1 size    | 256       |
| Hidden layer 2 size    | 128       |
| Batch size              | 64        |
| Learning rate            | 0.01      |
| Training iterations    | 300,000   |

## Project Structure

```
Next-Word-Predictor/
│
├── README.md                     # Project documentation
├── next_word_predictor.ipynb     # Main Colab notebook (data prep, training, UI)
│
└── Google Drive/ (Next_word_predictor/)
    ├── sentences.txt              # Training text corpus
    ├── embeddings.npy             # Saved word embedding table
    ├── model_parameters.pkl       # Saved network weights and biases
    └── vocab.pkl                  # Saved word-to-ID vocabulary mapping
```

> Note: The dataset and trained model files are stored on Google Drive (not in the GitHub repo) and are accessed by the notebook via the mounted Drive paths.

## Saved Artifacts

The following files are saved to / loaded from Google Drive so the model does not need to be retrained every session:

| File | Description | Path |
|------|-------------|------|
| `sentences.txt` | Training text corpus | `/content/drive/MyDrive/Next_word_predictor/sentences.txt` |
| `embeddings.npy` | Trained word embedding table | `/content/drive/MyDrive/Next_word_predictor/embeddings.npy` |
| `model_parameters.pkl` | Trained network weights and biases | `/content/drive/MyDrive/Next_word_predictor/model_parameters.pkl` |
| `vocab.pkl` | Word-to-ID vocabulary mapping | `/content/drive/MyDrive/Next_word_predictor/vocab.pkl` |

## Interactive Web Interface (Gradio)

The notebook includes a Gradio-based UI with the following features:

- **Live suggestions** – As the user types, the top 5 next-word predictions (with confidence percentages) are displayed and updated on every keystroke.
- **Click-to-complete** – Clicking a suggested word appends it to the text box and refreshes the suggestions.
- **Online learning** – A "Learn from this sentence" button fine-tunes the model (and embeddings) on the currently typed sentence using a small number of additional gradient steps, then saves the updated model back to Google Drive.
- **Custom styling** – The interface uses custom CSS for a cleaner, rounded-button look.

## Tech Stack

- Python
- NumPy (for all neural network math — no TensorFlow/PyTorch)
- Google Colab + Google Drive (for storage and execution)
- Gradio (for the web interface)
- `re` and `collections.Counter` (for tokenization and vocabulary building)
- `pickle` (for saving/loading parameters and vocabulary)

## How to Run

1. Open the notebook in Google Colab.
2. Mount Google Drive when prompted.
3. Ensure `sentences.txt` exists at the configured path on Drive (used as the training corpus).
4. Run all cells:
   - If saved model files (`embeddings.npy`, `model_parameters.pkl`, `vocab.pkl`) are found on Drive, they are loaded directly.
   - Otherwise, the model is trained from scratch and the resulting files are saved to Drive.
5. The final cells launch the Gradio app, producing a shareable link to interact with the next-word predictor.

## References

[1] Bengio, Y., Ducharme, R., Vincent, P., & Jauvin, C. (2003). A Neural Probabilistic Language Model. *Journal of Machine Learning Research*, 3, 1137–1155.

[2] NumPy Documentation. https://numpy.org/doc/

[3] Gradio Documentation. https://www.gradio.app/docs/

[4] Google Colab. https://colab.research.google.com/

[5] Glorot, X., & Bengio, Y. (2010). Understanding the difficulty of training deep feedforward neural networks. *AISTATS*.
