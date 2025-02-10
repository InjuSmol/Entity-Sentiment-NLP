# Sentiment Analysis for Target Entities in News Articles

<img width="992" alt="Screenshot 2025-02-09 at 6 57 07 PM" src="https://github.com/user-attachments/assets/9d136992-dae7-42b0-a7c8-76b361714386" />

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

## Future Improvements

- **Improve entity relevance detection** (filter out irrelevant text).
- **Incorporate context-based sentiment weighting**.
- **Use ensemble models** for better sentiment classification.

## Contributors

- Ethan Yen

- Zoheb Hasan

- InjuSmol

## License

MIT License. See `LICENSE` for details.

