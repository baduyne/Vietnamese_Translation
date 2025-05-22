# English-to-Vietnamese Neural Machine Translation

## 📌 Overview

This project focuses on building a **Neural Machine Translation (NMT)** system that translates **English** sentences to **Vietnamese** using **Transformer architecture**, implemented **from scratch**. The goal is to explore the effectiveness of **Natural Language Processing (NLP)** techniques in machine translation, with full control over each training step.

## 📂 Dataset

The dataset was collected and curated from two primary sources:

- [Tatoeba](https://tatoeba.org/): An open collection of multilingual sentences and translations.
- [OPUS](https://opus.nlpl.eu/): A repository of parallel corpora for machine translation.

After crawling, a substantial parallel corpus of English-Vietnamese sentence pairs was obtained for training and evaluation.

### 🔧 Data Preprocessing

The raw data underwent several cleaning steps:

- **Lowercasing** all texts.
- **Removing punctuation**, non-printable characters, and HTML tags.
- **Filtering out** sentence pairs that are too long or too short (e.g., fewer than 3 words or more than 50 words).
- **Deduplication** and **removal of misaligned pairs** (e.g., when one sentence is empty or mismatched).
- Normalizing Vietnamese diacritics using [pyvi](https://pypi.org/project/pyvi/) and Unicode NFC form.

After cleaning, the dataset was split into training, validation, and testing sets.

## 🧠 Model Architecture

The core model is a **Transformer** inspired by *"Attention is All You Need"* (Vaswani et al., 2017), implemented step by step using PyTorch (or TensorFlow depending on your implementation). The key components include:

- Positional Encoding
- Multi-Head Self-Attention
- Encoder & Decoder blocks
- Masked attention for decoder inputs
- Layer Normalization & Residual Connections
- Final Linear + Softmax output layer

### 🧪 Features

- **Trained from scratch** without using high-level libraries like Hugging Face.
- Modular, readable implementation.
- Custom training loop with teacher forcing.
- Training with **cross-entropy loss** and **Adam optimizer**.
- Support for **early stopping** and model checkpointing.

## ✂️ Tokenization

Instead of using pre-trained tokenizers, we **built a dictionary directly from the dataset**:

- Word-level tokenization for both English and Vietnamese.
- Special tokens: `<pad>`, `<sos>`, `<eos>`, `<unk>`.
- Vocabulary built based on word frequency (with an adjustable frequency threshold).
- Tokenization and detokenization handled through custom `Tokenizer` class.

## 📈 Evaluation

Model performance is evaluated using:

- **BLEU score** on test set.
- Qualitative evaluation by manually reviewing translations.

You can run:

```bash
python evaluate.py --model-path saved_model.pt
