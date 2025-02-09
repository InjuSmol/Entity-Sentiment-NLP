# Sentiment Analysis for Target Entities in News Articles

## Overview

This project focuses on training and evaluating NLP models for **targeted sentiment analysis** in news articles. The goal is to classify the sentiment of a given **target entity** mentioned in a document as **positive, negative, or neutral**.

## Problem Statement

**Traditional sentiment analysis** classifies entire documents or sentences, but this approach is insufficient when multiple entities are mentioned in a single text. This project aims to develop a **targeted sentiment classification model** to determine how an entity is perceived in a given article.

### Example

#### Input:

> "The new movie *Avengers* has fantastic CGI but a contrived plot."

#### Output:

> **Entity:** *Avengers* → **Neutral Sentiment**

## Approach

### Models Used:

- **Fine-tuned BERT**
- **Fine-tuned DistilBERT** (lighter version of BERT)
- **Fine-tuned RoBERTa** (enhanced pretraining for better language understanding)

### Steps:

1. **Data Preprocessing:**
   - Extract **target entities** from documents.
   - Tokenize text using the **Hugging Face Tokenizer**.
   - Label data for supervised learning.
2. **Model Training:**
   - Fine-tune **BERT, DistilBERT, and RoBERTa**.
   - Train models with **multi-class classification** (positive, negative, neutral).
3. **Evaluation:**
   - Use **accuracy, precision, recall, and F1-score**.
   - Compare models on **random and fixed test sets**.

## Dataset

- **Source:** PerSent Dataset
- **Split:**
  - Training Set: **3,355 examples**
  - Validation Set: **578 examples**
  - Test Set: **579 (random) + 827 (fixed) examples**

## Results Summary

| Model      | Accuracy (Random) | Accuracy (Fixed) |
| ---------- | ----------------- | ---------------- |
| BERT       | 57.3%             | 49.0%            |
| DistilBERT | 58.0%             | 47.6%            |
| RoBERTa    | 53.7%             | 46.0%            |

**Observations:**

- **BERT performed best** but was computationally expensive.
- **DistilBERT was nearly as good** while being faster and more efficient.
- **RoBERTa underperformed**, struggling with mixed sentiments.

## Installation

### Prerequisites:

- Python 3.8+
- PyTorch
- Hugging Face Transformers
- Scikit-learn
- Pandas, NumPy

### Setup:

```sh
# Clone the repository
git clone https://github.com/your-repo/sentiment-analysis.git
cd sentiment-analysis

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Training a Model:

```sh
python train.py --model bert --epochs 5
```

### Running Inference:

```sh
python predict.py --text "The new Tesla Model Y is amazing!" --entity "Tesla"
```

### Evaluating Models:

```sh
python evaluate.py --model distilbert
```

## Future Improvements

- **Improve entity relevance detection** (filter out irrelevant text).
- **Incorporate context-based sentiment weighting**.
- **Use ensemble models** for better sentiment classification.

## Contributors

- **Your Name**
- [GitHub](https://github.com/your-profile)
- [Email](mailto\:your.email@example.com)

## License

MIT License. See `LICENSE` for details.

