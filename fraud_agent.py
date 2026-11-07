"""
Fraud Detection Agent - model + analysis

Dataset : ULB / Dal Pozzolo credit card fraud (Kaggle: mlg-ulb/creditcardfraud)
Model   : Random Forest (supervised, ensemble), class_weight="balanced"
Output  : charts in outputs/, every metric printed AND saved to outputs/metrics.json

"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

ROOT = Path(__file__).parent
DATA = ROOT / "data" / "creditcard.csv"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

SEED = 42
sns.set_theme(style="whitegrid")
# Small base fonts: charts are sized to match their slots in the slide deck (10in wide slides)
plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
                     "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 8})
LEGIT_C, FRAUD_C = "#2f6fed", "#e0457b"
metrics = {}

# --------------------------------------------------------------------------
# 1. LOAD + BASIC FACTS  (Slide 3)
# --------------------------------------------------------------------------
df = pd.read_csv(DATA)
n_rows, n_cols = df.shape
n_fraud = int(df["Class"].sum())
n_legit = n_rows - n_fraud
metrics["data"] = {
    "rows": n_rows,
    "columns": n_cols,
    "fraud": n_fraud,
    "legit": n_legit,
    "fraud_pct": round(100 * n_fraud / n_rows, 4),
    "missing_values": int(df.isna().sum().sum()),
    "duplicate_rows": int(df.duplicated().sum()),
    "amount_min": float(df["Amount"].min()),
    "amount_max": float(df["Amount"].max()),
    "amount_mean": round(float(df["Amount"].mean()), 2),
    "amount_median": round(float(df["Amount"].median()), 2),
    "amount_mean_fraud": round(float(df.loc[df.Class == 1, "Amount"].mean()), 2),
    "amount_mean_legit": round(float(df.loc[df.Class == 0, "Amount"].mean()), 2),
    "time_span_hours": round(float(df["Time"].max() / 3600), 1),
}
print("=== DATA ===")
print(json.dumps(metrics["data"], indent=2))

# --------------------------------------------------------------------------
# 2. EDA CHARTS  (Slide 4)
# --------------------------------------------------------------------------
# 2a. Class balance
fig, ax = plt.subplots(figsize=(4.1, 2.6))
bars = ax.bar(["Legit (0)", "Fraud (1)"], [n_legit, n_fraud], color=[LEGIT_C, FRAUD_C])
ax.set_yscale("log")
ax.set_ylim(1, 3e6)
ax.set_ylabel("Transactions (log scale)")
ax.set_title("Class Balance - Extremely Imbalanced")
for b, v in zip(bars, [n_legit, n_fraud]):
    ax.text(b.get_x() + b.get_width() / 2, v * 1.3, f"{v:,}", ha="center", fontweight="bold")
ax.text(0.75, 0.80, f"Fraud = {metrics['data']['fraud_pct']}%\n(about 1 in {round(n_rows / n_fraud)})",
        transform=ax.transAxes, ha="center", fontsize=9, color=FRAUD_C, fontweight="bold")
fig.tight_layout()
fig.savefig(OUT / "01_class_balance.png", dpi=220)
plt.close(fig)

# 2b. Amount distribution, fraud vs legit
fraud_amt = df.loc[df.Class == 1, "Amount"]
legit_amt = df.loc[df.Class == 0, "Amount"]
metrics["amount_insight"] = {
    "fraud_median": round(float(fraud_amt.median()), 2),
    "legit_median": round(float(legit_amt.median()), 2),
    "fraud_share_le_1_eur_pct": round(100 * float((fraud_amt <= 1).mean()), 1),
    "legit_share_le_1_eur_pct": round(100 * float((legit_amt <= 1).mean()), 1),
    "fraud_zero_amount_count": int((fraud_amt == 0).sum()),
}
# log10(1 + Amount) keeps EUR 0 at 0 instead of squashing it into one fake bin
fig, ax = plt.subplots(figsize=(4.6, 1.9))
bins = np.linspace(0, np.log10(df["Amount"].max() + 1), 45)
ax.hist(np.log10(1 + legit_amt), bins=bins, alpha=0.6, color=LEGIT_C, label="Legit", density=True)
ax.hist(np.log10(1 + fraud_amt), bins=bins, alpha=0.6, color=FRAUD_C, label="Fraud", density=True)
tick_vals = [0, 10, 100, 1000, 10000]
ax.set_xticks([np.log10(1 + v) for v in tick_vals])
ax.set_xticklabels([f"{v:,}" for v in tick_vals])
ax.set_xlabel("Transaction amount (EUR, log scale)")
ax.set_ylabel("Density")
ax.set_title("Amount Distribution: Fraud vs Legit")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "02_amount_distribution.png", dpi=220)
plt.close(fig)

# 2c. Correlation with the target + heatmap of the strongest features
corr = df.corr()
target_corr = corr["Class"].drop("Class")
top_feats = target_corr.abs().sort_values(ascending=False).head(10).index.tolist()
metrics["top_correlated_with_fraud"] = {f: round(float(target_corr[f]), 3) for f in top_feats}

# Heatmap: Time, Amount + 6 strongest V features + Class.
# (V1..V28 are PCA outputs, so they are uncorrelated with each other by design;
#  Time/Amount are the only raw fields and DO correlate with some V's.)
heat_cols = ["Time", "Amount"] + [f for f in top_feats if f.startswith("V")][:4] + ["Class"]
fig, ax = plt.subplots(figsize=(4.2, 3.6))
sns.heatmap(df[heat_cols].corr(), annot=True, fmt=".2f", cmap="RdBu_r",
            center=0, vmin=-1, vmax=1, ax=ax, annot_kws={"size": 8}, cbar_kws={"shrink": 0.8})
ax.set_title("Correlation Heatmap")
fig.tight_layout()
fig.savefig(OUT / "03_correlation_heatmap.png", dpi=220)
plt.close(fig)

# Bar chart: correlation of EVERY feature with fraud
tc = target_corr.sort_values()
fig, ax = plt.subplots(figsize=(7, 7))
ax.barh(tc.index, tc.values, color=[FRAUD_C if v > 0 else LEGIT_C for v in tc.values])
ax.axvline(0, color="black", lw=0.8)
ax.set_xlabel("Correlation with fraud (Class)")
ax.set_title("Which Features Move With Fraud?")
ax.tick_params(axis="y", labelsize=8)
fig.tight_layout()
fig.savefig(OUT / "03b_correlation_with_fraud.png", dpi=220)
plt.close(fig)

# 2d. Fraud vs legit boxplots for the top 4 features
top3 = top_feats[:3]
fig, axes = plt.subplots(1, 3, figsize=(4.6, 1.6))
for ax, f in zip(axes, top3):
    sns.boxplot(data=df, x="Class", y=f, hue="Class", palette=[LEGIT_C, FRAUD_C],
                legend=False, ax=ax, fliersize=1)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Legit", "Fraud"])
    ax.set_xlabel("")
    ax.set_title(f"{f}  (r = {target_corr[f]:+.2f})")
fig.tight_layout()
fig.savefig(OUT / "04_top_feature_boxplots.png", dpi=200)
plt.close(fig)

# 2e. Fraud over time (fraud rate per hour of the 2 days)
hour = (df["Time"] // 3600).astype(int)
by_hour = df.groupby(hour)["Class"].agg(["sum", "count"])
by_hour["rate_pct"] = 100 * by_hour["sum"] / by_hour["count"]
fig, ax1 = plt.subplots(figsize=(8, 4.5))
ax1.bar(by_hour.index, by_hour["count"], color=LEGIT_C, alpha=0.5, label="Transactions")
ax1.set_xlabel("Hours since first transaction")
ax1.set_ylabel("Transactions per hour")
ax2 = ax1.twinx()
ax2.plot(by_hour.index, by_hour["rate_pct"], color=FRAUD_C, marker="o", ms=3, label="Fraud rate %")
ax2.set_ylabel("Fraud rate (%)")
ax2.grid(False)
ax1.set_title("Volume vs Fraud Rate Over the 2 Days")
fig.tight_layout()
fig.savefig(OUT / "05_fraud_over_time.png", dpi=220)
plt.close(fig)

# --------------------------------------------------------------------------
# 3. PREPROCESS + SPLIT  (Slide 5)
# --------------------------------------------------------------------------
X = df.drop(columns="Class").copy()
y = df["Class"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=SEED
)

# Scale Time + Amount (fit on TRAIN only -> no leakage). V1..V28 are already PCA-scaled.
# Note: Random Forest does not need scaling; done for good practice / consistency.
scaler = RobustScaler()
X_train[["Time", "Amount"]] = scaler.fit_transform(X_train[["Time", "Amount"]])
X_test[["Time", "Amount"]] = scaler.transform(X_test[["Time", "Amount"]])

metrics["split"] = {
    "train_rows": len(X_train),
    "test_rows": len(X_test),
    "train_fraud": int(y_train.sum()),
    "test_fraud": int(y_test.sum()),
    "test_size": 0.25,
    "stratified": True,
}
print("\n=== SPLIT ===")
print(json.dumps(metrics["split"], indent=2))

# --------------------------------------------------------------------------
# 4. TRAIN RANDOM FOREST  (Slide 5)
# --------------------------------------------------------------------------
params = dict(
    n_estimators=200,
    max_depth=12,
    min_samples_leaf=2,
    class_weight="balanced",
    n_jobs=-1,
    random_state=SEED,
)
metrics["hyperparameters"] = {k: v for k, v in params.items() if k != "n_jobs"}
print("\n=== TRAINING Random Forest ===", params)
rf = RandomForestClassifier(**params)
rf.fit(X_train, y_train)

# --------------------------------------------------------------------------
# 5. EVALUATE ON THE UNTOUCHED TEST SET  (Slide 6)
# --------------------------------------------------------------------------
proba = rf.predict_proba(X_test)[:, 1]
THRESH = 0.5
pred = (proba >= THRESH).astype(int)

tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
metrics["results"] = {
    "threshold": THRESH,
    "accuracy": round(float((pred == y_test).mean()), 5),
    "precision_fraud": round(float(precision_score(y_test, pred)), 4),
    "recall_fraud": round(float(recall_score(y_test, pred)), 4),
    "f1_fraud": round(float(f1_score(y_test, pred)), 4),
    "roc_auc": round(float(roc_auc_score(y_test, proba)), 4),
    "pr_auc_average_precision": round(float(average_precision_score(y_test, proba)), 4),
    "confusion": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)},
    "baseline_accuracy_if_always_legit": round(float((y_test == 0).mean()), 5),
}
print("\n=== RESULTS (test set) ===")
print(json.dumps(metrics["results"], indent=2))
print("\n", classification_report(y_test, pred, target_names=["Legit", "Fraud"], digits=4))

# Threshold sweep -> "how could we make it better" (recall/precision trade-off)
sweep = {}
for t in [0.2, 0.3, 0.4, 0.5, 0.6]:
    p = (proba >= t).astype(int)
    sweep[str(t)] = {
        "precision": round(float(precision_score(y_test, p, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, p)), 4),
        "f1": round(float(f1_score(y_test, p)), 4),
        "missed_fraud_FN": int(((p == 0) & (y_test == 1)).sum()),
        "false_alarms_FP": int(((p == 1) & (y_test == 0)).sum()),
    }
metrics["threshold_sweep"] = sweep
print("=== THRESHOLD SWEEP ===")
print(json.dumps(sweep, indent=2))

# Confusion matrix chart
fig, ax = plt.subplots(figsize=(3.0, 2.9))
ConfusionMatrixDisplay(confusion_matrix(y_test, pred), display_labels=["Legit", "Fraud"]).plot(
    ax=ax, cmap="Blues", colorbar=False, values_format=",d", text_kw={"fontsize": 9})
ax.set_title("Confusion Matrix (Test Set)")
ax.grid(False)
fig.tight_layout()
fig.savefig(OUT / "06_confusion_matrix.png", dpi=220)
plt.close(fig)

# Feature importance chart
imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
metrics["top_feature_importances"] = {k: round(float(v), 4) for k, v in imp.head(8).items()}
fig, ax = plt.subplots(figsize=(3.4, 2.5))
imp.head(8).iloc[::-1].plot.barh(ax=ax, color=FRAUD_C)
ax.set_xlabel("Importance")
ax.set_title("Top 8 Feature Importances (Random Forest)")
fig.tight_layout()
fig.savefig(OUT / "07_feature_importance.png", dpi=220)
plt.close(fig)

# Precision-Recall curve
prec, rec, _ = precision_recall_curve(y_test, proba)
fig, ax = plt.subplots(figsize=(5.5, 4.5))
ax.plot(rec, prec, color=FRAUD_C, lw=2)
ax.set_xlabel("Recall (fraud caught)")
ax.set_ylabel("Precision (alerts that were real fraud)")
ax.set_title(f"Precision-Recall Curve (AP = {metrics['results']['pr_auc_average_precision']:.3f})")
fig.tight_layout()
fig.savefig(OUT / "08_precision_recall_curve.png", dpi=220)
plt.close(fig)

# --------------------------------------------------------------------------
# 6. SAVE EVERYTHING
# --------------------------------------------------------------------------
(OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
print("\nSaved charts + metrics.json to", OUT)
