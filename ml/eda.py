from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw straight to image files, no pop-up window

import matplotlib.pyplot as plt
import seaborn as sns

from ml.common import FEATURES, RISK_ORDER, load_data

OUTPUT_DIR = Path(__file__).parent / "eda_output"
OUTPUT_DIR.mkdir(exist_ok=True)

df = load_data()

# ---- 1. Basic checks -------------------------------------------------------
print("Shape (rows, columns):", df.shape)
print("\nFirst 5 rows:")
print(df.head())

print("\nMissing values per column:")
print(df.isnull().sum())

print("\nSummary statistics:")
print(df[FEATURES].describe().round(2))

print("\nClass counts:")
print(df["risk_level"].value_counts().reindex(RISK_ORDER))

# ---- 2. Chart: class balance ----------------------------------------------
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x="risk_level", order=RISK_ORDER)
plt.title("Number of students per risk level")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "1_class_distribution.png", dpi=120)
plt.close()

# ---- 3. Charts: each feature split by risk level --------------------------
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for ax, feature in zip(axes.flat, FEATURES):
    sns.boxplot(data=df, x="risk_level", y=feature, order=RISK_ORDER, ax=ax)
    ax.set_title(f"{feature} by risk level")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "2_features_by_risk.png", dpi=120)
plt.close()

# ---- 4. Chart: how features relate to each other --------------------------
plt.figure(figsize=(8, 6))
sns.heatmap(df[FEATURES].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Correlation between features")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "3_correlation.png", dpi=120)
plt.close()

print(f"\nCharts saved in: {OUTPUT_DIR}")