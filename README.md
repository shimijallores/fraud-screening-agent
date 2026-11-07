# Fraud Screening Agent

A small AI agent that screens card transactions and flags likely fraud for review.
It uses a Random Forest trained on the [ULB Credit Card Fraud dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) (284,807 real transactions, 0.17% fraud).

Built for the AI course retake (Software Engineering, SoSe26).

## Features

- Data exploration: class balance, amount distribution, correlation heatmap, fraud vs legit boxplots
- Random Forest classifier (`class_weight="balanced"` for the fraud imbalance)
- Honest evaluation on an untouched test set: recall, precision, F1, ROC-AUC, confusion matrix, feature importance
- All results saved to `outputs/` (charts and `metrics.json`)
- 7-slide presentation built from those results, so every number is real

## How to Run

1. Create an environment and install the libraries:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. Download the dataset into `data/` (needs the [Kaggle CLI](https://github.com/Kaggle/kaggle-api) set up), or download it manually from the link above:

   ```bash
   kaggle datasets download -d mlg-ulb/creditcardfraud -p data --unzip
   ```

3. Train the model and generate the charts and metrics:

   ```bash
   python fraud_agent.py
   ```

Output: `outputs/`
