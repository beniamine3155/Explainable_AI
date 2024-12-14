# Quora Insincere Questions Classification usin LIME

This repository contains a Python implementation of a text classification pipeline for detecting insincere questions on Quora. The project demonstrates the use of **Logistic Regression**, **TF-IDF Vectorization**, and **Evaluation Metrics** to classify text data. It also explores model interpretability using **LIME (Local Interpretable Model-agnostic Explanations)**.

## Features

- **Dataset Handling**: Preprocessing of the Quora Insincere Questions dataset to remove missing values and split it into training and validation sets.
- **TF-IDF Vectorization**: Conversion of text into numerical features using Term Frequency-Inverse Document Frequency (TF-IDF) with bi-gram analysis.
- **Classification Model**: Logistic Regression for binary classification of sincere vs. insincere questions.
- **Evaluation**: Includes metrics such as accuracy, precision, recall, F1 score, confusion matrix, and classification report.
- **Error Analysis**: Identification and inspection of misclassified samples.
- **Interpretability**: Insights into predictions using the LIME library.

## Requirements

Install the required Python libraries using:

```bash
pip install lime
```

### Main Libraries

- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `lime`

## How to Run

1. Clone this repository:

```bash
git clone https://github.com/beniamine3155/Explainable_AI/tree/main/LIME


4. Analyze the output metrics and confusion matrix displayed in the console.

## Dataset

The dataset used in this project is the **Quora Insincere Questions Classification** dataset. Ensure you have the `train.csv` file in the `data` folder.

- Download the dataset from [Kaggle](https://www.kaggle.com/competitions/quora-insincere-questions-classification/data).
- Place the dataset in the `data` directory.

## Results

### Performance Metrics

| Metric          | Value         |
|-----------------|---------------|
| Accuracy        | 94.6%         |
| Precision       | 67.6%         |
| Recall          | 25.2%         |
| F1 Score        | 36.8%         |

### Confusion Matrix

```

[[18915   161]
 [ 1065   359]]

```

### Insights
- The model achieves high accuracy but struggles with identifying insincere questions due to class imbalance.
- Precision for insincere questions is moderate, but recall is relatively low, indicating missed insincere questions.

## Future Improvements

- **Class Imbalance**: Address the imbalance using oversampling, undersampling, or class weighting techniques.
- **Advanced Models**: Experiment with Random Forest, XGBoost, or neural networks for improved performance.
- **Feature Engineering**: Include additional features like question length, punctuation, or embeddings (GloVe, Word2Vec).
- **Interpretability**: Use LIME to refine predictions and better understand the model's decisions.


```
