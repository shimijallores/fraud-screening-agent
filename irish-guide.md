# Irish: Review and Presentation Guide

## The project in 3 lines

- You built an AI agent that screens card transactions and flags likely fraud.
- It uses a **Random Forest** (200 decision trees that vote), trained on 284,807 real transactions from Kaggle.

## Numbers to remember

| What          | Value                                                            |
| ------------- | ---------------------------------------------------------------- |
| Dataset       | Kaggle, ULB Credit Card Fraud (284,807 transactions, 2 days)     |
| Fraud rate    | 0.17% (492 frauds, about 1 in 579)                               |
| Split         | 75% train / 25% test (71,202 test rows, 123 fraud)               |
| **Recall**    | **81.3%** (caught 100 of 123 frauds)                             |
| **Precision** | **77.5%** (100 of 129 alerts were real fraud)                    |
| F1 / ROC-AUC  | 0.794 / 0.976                                                    |
| Mistakes      | 23 missed frauds, 29 real transactions wrongly flagged           |
| Accuracy      | 99.93%, but "always say legit" scores 99.83%, so it means little |

## Words you must be able to explain

- **Supervised learning:** the model learns from examples that already have the answer (fraud or legit).
- **Random Forest:** many decision trees each vote, and the votes are averaged into a fraud probability.
- **Imbalanced data:** fraud is rare, so a model can look accurate by ignoring it.
- **class_weight = "balanced":** missing a fraud counts more during training. No fake data is created.
- **Recall:** of all real frauds, how many we caught.
- **Precision:** of all alerts, how many were real fraud.
- **Threshold (0.5):** the probability needed to flag a transaction. Lower means more caught, more false alarms.
- **V1 to V28:** anonymized features. The dataset hides what they mean for privacy.

## What to say, slide by slide

1. **Title.** "This is a fraud screening agent. It flags suspicious card transactions for review."
2. **Environment (30 seconds max).** Read the table. Key line: "Partially observable, stochastic, episodic, dynamic, mostly discrete, multi-agent."
3. **Data.** Say where it came from (Kaggle), what it is, and the problems: only 0.17% fraud, 1,081 duplicate rows, anonymized features.
4. **Data inspection.** "V17, V14 and V12 separate fraud best. Also 36.8% of frauds are EUR 1 or less, against 10.7% of legit."
5. **Model.** Point to the diagram: input, trees, vote, probability. Name the settings: 200 trees, depth 12, balanced class weight, threshold 0.5.
6. **Results.** Lead with recall and precision, not accuracy. Explain why accuracy misleads. Then give the improvement: a 0.3 threshold raises recall to 87.0% but causes 86 false alarms.
7. **Agent.** "Perceive, predict, decide, act. The model is one step. It is a learning agent while training and a simple reflex agent when running."

## Likely questions

- **Why not just report accuracy?** Because 99.83% is possible by always saying "legit". Recall and precision show whether fraud is caught.
- **Why Random Forest?** It finds combinations of features, handles imbalance with class weights, and shows which features matter.
- **Why not SMOTE?** It invents fake fraud rows. `class_weight` keeps the data real.
- **Is it a reflex or learning agent?** Learning while training. Once running, it applies a fixed rule: score ≥ 0.5, flag.
- **What can't it do?** It can't explain why (features are anonymized), can't see intent, and doesn't learn after training.
- **How would you improve it?** Tune settings, lower the threshold, add more fraud examples, retrain on new data, add a feedback loop.
- **Did you test on training data?** No. The test set was held out, and the scaler was fit on training data only.
- **Why did you change from GCash?** The professor said the same kind of topic was fine if I found a real dataset. Kaggle had fraud data.
- **Does it use an online API?** No. The model is trained and runs locally with scikit-learn.
- **Is it overfitting?** I only used one train/test split. Next I would use cross-validation.
- **Why are fraud amounts small?** The data shows it. A common explanation is "card testing", but that is a guess, not proven here.

## Do not say

- "It is 99.9% accurate."
- "V14 means ..." (nobody knows what the anonymized features are).
- "It keeps learning from new transactions." (It doesn't.)
