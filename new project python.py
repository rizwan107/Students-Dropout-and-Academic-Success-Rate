import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind, chi2_contingency, norm
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# Set style
sns.set_style("white")
plt.rcParams["font.size"] = 12
plt.rcParams["font.family"] = "Arial"
plt.rcParams["axes.facecolor"] = "white"
plt.rcParams["figure.facecolor"] = "#f5f5f5"

# Load dataset
df = pd.read_csv(r"C:\Users\AUSU\Downloads\data.csv")

# Curricular columns
curricular_cols = [
    "Curricular units 1st sem (credited)", "Curricular units 1st sem (enrolled)",
    "Curricular units 1st sem (evaluations)", "Curricular units 1st sem (approved)",
    "Curricular units 1st sem (grade)", "Curricular units 1st sem (without evaluations)",
    "Curricular units 2nd sem (credited)", "Curricular units 2nd sem (enrolled)",
    "Curricular units 2nd sem (evaluations)", "Curricular units 2nd sem (approved)",
    "Curricular units 2nd sem (grade)", "Curricular units 2nd sem (without evaluations)"
]

# Short names
short_names = {
    "Admission grade": "Admission", "Curricular units 1st sem (credited)": "1st Credited",
    "Curricular units 1st sem (enrolled)": "1st Enrolled", "Curricular units 1st sem (evaluations)": "1st Evaluations",
    "Curricular units 1st sem (approved)": "1st Approved", "Curricular units 1st sem (grade)": "1st Grade",
    "Curricular units 1st sem (without evaluations)": "1st No Eval", "Curricular units 2nd sem (credited)": "2nd Credited",
    "Curricular units 2nd sem (enrolled)": "2nd Enrolled", "Curricular units 2nd sem (evaluations)": "2nd Evaluations",
    "Curricular units 2nd sem (approved)": "2nd Approved", "Curricular units 2nd sem (grade)": "2nd Grade",
    "Curricular units 2nd sem (without evaluations)": "2nd No Eval"
}

# --- Dropout and Success Rates ---
print("\n=== Dropout and Academic Success Rates ===")
outcome_counts = df["Target"].value_counts(normalize=True) * 100
print("Outcome Percentages:")
print(outcome_counts.round(2))
dropout_rate = outcome_counts["Dropout"]
success_rate = outcome_counts["Graduate"] + outcome_counts["Enrolled"]
print(f"Dropout Rate: {dropout_rate:.2f}%")
print(f"Success Rate (Graduate + Enrolled): {success_rate:.2f}%")

# --- EDA ---
print("\n=== EDA for Dropout and Success Analysis ===")
print("Shape:", df.shape)
print("\nData Types:\n", df.dtypes[["Admission grade", "Target", *curricular_cols]])
print("\nDuplicates:", df.duplicated().sum())
print("\nNulls:\n", df[["Admission grade", "Target", *curricular_cols]].isnull().sum())

# --- Group Stats ---
print("\n=== Group Stats for Dropout vs. Success ===")
print("\nAdmission Grade by Outcome:")
print(df.groupby("Target")["Admission grade"].agg(["mean", "median"]).round(2))
for col in curricular_cols:
    print(f"\n{col} by Outcome:")
    print(df.groupby("Target")[col].agg(["mean", "median"]).round(2))

# --- Scholarship Analysis ---
print("\n=== Scholarship Impact on Dropout/Success ===")
print("\nScholarship Holder Counts:")
print(df["Scholarship holder"].value_counts())
contingency = pd.crosstab(df["Scholarship holder"], df["Target"])
print("\nContingency Table:")
print(contingency)
contingency_pct = pd.crosstab(df["Scholarship holder"], df["Target"], normalize="index") * 100
print("\nPercentage by Scholarship Status:")
print(contingency_pct.round(2))
chi2, p, _, _ = chi2_contingency(contingency)
print(f"Chi-squared: chi2={chi2:.2f}, p-value={p:.4f}")

# --- T-tests ---
print("\n=== T-tests: Graduate vs. Dropout ===")
t_results = []
for col in curricular_cols:
    grad_vals = df[df["Target"] == "Graduate"][col]
    drop_vals = df[df["Target"] == "Dropout"][col]
    t_stat, p_val = ttest_ind(grad_vals, drop_vals)
    mean_diff = grad_vals.mean() - drop_vals.mean()
    se_diff = np.sqrt(grad_vals.var()/len(grad_vals) + drop_vals.var()/len(drop_vals))
    ci_95 = (mean_diff - 1.96 * se_diff, mean_diff + 1.96 * se_diff)
    t_results.append({"Column": col, "t_stat": t_stat, "p_val": p_val, "mean_diff": mean_diff, "ci_lower": ci_95[0], "ci_upper": ci_95[1]})
print(f"{'Metric':<25} {'t-stat':>8} {'p-value':>10} {'Mean Diff':>12} {'95% CI Lower':>12} {'95% CI Upper':>12}")
for res in t_results:
    print(f"{short_names[res['Column']]:<25} {res['t_stat']:>8.2f} {res['p_val']:>10.4f} {res['mean_diff']:>12.2f} {res['ci_lower']:>12.2f} {res['ci_upper']:>12.2f}")

# --- Correlation ---
print("\n=== Correlations Impacting Success ===")
key_curricular_cols = ["Curricular units 1st sem (approved)", "Curricular units 1st sem (grade)",
                       "Curricular units 2nd sem (approved)", "Curricular units 2nd sem (grade)"]
corr_cols = ["Admission grade"] + key_curricular_cols
corr = df[corr_cols].corr()
corr.index = [short_names.get(col, col) for col in corr_cols]
corr.columns = [short_names.get(col, col) for col in corr_cols]
print("Correlation Matrix:\n", corr.round(2))

# --- Charts ---
# 1. Box Plot
fig, ax = plt.subplots(figsize=(10, 7))
sns.boxplot(x="Target", y="Admission grade", hue="Target", data=df, palette="Set3", ax=ax, legend=False)
ax.set_title("Admission Grades by Outcome", fontsize=16, weight="bold", pad=10)
ax.set_xlabel("Outcome", fontsize=14)
ax.set_ylabel("Admission Grade", fontsize=14)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
for i, target in enumerate(df["Target"].unique()):
    median_val = df[df["Target"] == target]["Admission grade"].median()
    ax.annotate(f"Median: {median_val:.1f}", xy=(i, median_val), xytext=(i, median_val + 3),
                ha="center", fontsize=8, bbox=dict(boxstyle="round,pad=0.3", edgecolor="gray", facecolor="white"))
ax.annotate("Chart: Box Plot", xy=(0.02, 0.90), xycoords="axes fraction", fontsize=8, color="navy",
            bbox=dict(boxstyle="round,pad=0.3", edgecolor="navy", facecolor="white"))
plt.tight_layout(pad=1.5)
plt.savefig("box_plot.png")
plt.show()

# 2. Stacked Bar Plot
contingency_pct = pd.crosstab(df["Scholarship holder"], df["Target"], normalize="index") * 100
fig, ax = plt.subplots(figsize=(10, 7))
contingency_pct.plot(kind="bar", stacked=True, color=["#FF6B6B", "#40C4FF", "#FFD700"], ax=ax, edgecolor="white", linewidth=1.5)
ax.set_title("Outcome by Scholarship Status", fontsize=16, weight="bold", pad=10)
ax.set_xlabel("Scholarship Holder (0 = No, 1 = Yes)", fontsize=14)
ax.set_ylabel("Percentage (%)", fontsize=14)
ax.legend(title="Outcome", loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3, fontsize=12)
for i, row in enumerate(contingency_pct.values):
    cumsum = 0
    for j, val in enumerate(row):
        if val > 0:
            cumsum += val / 2
            ax.text(i, cumsum, f"{val:.1f}%", ha="center", va="center", color="white", fontsize=8, weight="bold")
            cumsum += val / 2
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.annotate("Chart: Stacked Bar Plot", xy=(0.02, 0.90), xycoords="axes fraction", fontsize=8, color="navy",
            bbox=dict(boxstyle="round,pad=0.3", edgecolor="navy", facecolor="white"))
plt.tight_layout(pad=1.5)
plt.savefig("stacked_bar.png")
plt.show()

# 3. Histogram: 1st Sem Approved
fig, ax = plt.subplots(figsize=(10, 7))
sns.histplot(data=df, x="Curricular units 1st sem (approved)", bins=20, color="lightblue", 
             alpha=0.7, ax=ax, edgecolor="black", stat="count")
# Add KDE (Kernel Density Estimate) line for wave effect
sns.kdeplot(data=df, x="Curricular units 1st sem (approved)", color="blue", ax=ax, lw=2)
ax.set_title("Distribution of 1st Semester Approved Units", fontsize=16, weight="bold", pad=10)
ax.set_xlabel("Approved Units", fontsize=14)
ax.set_ylabel("Count", fontsize=14)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.annotate("Chart: Histogram", xy=(0.02, 0.90), xycoords="axes fraction", fontsize=8, color="navy",
            bbox=dict(boxstyle="round,pad=0.3", edgecolor="navy", facecolor="white"))
plt.tight_layout(pad=1.5)
plt.savefig("hist_1st_approved.png")
plt.show()

# 4. Histogram: 2nd Sem Approved
fig, ax = plt.subplots(figsize=(10, 7))
sns.histplot(data=df, x="Curricular units 2nd sem (approved)", bins=20, color="lightblue", 
             alpha=0.7, ax=ax, edgecolor="black", stat="count")
# Add KDE (Kernel Density Estimate) line for wave effect
sns.kdeplot(data=df, x="Curricular units 2nd sem (approved)", color="blue", ax=ax, lw=2)
ax.set_title("Distribution of 2nd Semester Approved Units", fontsize=16, weight="bold", pad=10)
ax.set_xlabel("Approved Units", fontsize=14)
ax.set_ylabel("Count", fontsize=14)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.annotate("Chart: Histogram", xy=(0.02, 0.90), xycoords="axes fraction", fontsize=8, color="navy",
            bbox=dict(boxstyle="round,pad=0.3", edgecolor="navy", facecolor="white"))
plt.tight_layout(pad=1.5)
plt.savefig("hist_2nd_approved.png")
plt.show()

# 5. Heatmap: Mean Differences
mean_diff_data = pd.DataFrame({
    "Metric": [short_names[col] for col in curricular_cols],
    "Mean Difference": [df[df["Target"] == "Graduate"][col].mean() - df[df["Target"] == "Dropout"][col].mean() for col in curricular_cols]
}).set_index("Metric").sort_values("Mean Difference", ascending=False)
fig, ax = plt.subplots(figsize=(10, 7))
sns.heatmap(mean_diff_data, annot=True, cmap="YlGnBu", cbar_kws={"label": "Mean Difference"}, annot_kws={"size": 10})
ax.set_title("Mean Differences: Graduate vs. Dropout", fontsize=16, weight="bold", pad=10)
ax.set_ylabel("")
ax.annotate("Chart: Heatmap", xy=(0.02, 0.90), xycoords="axes fraction", fontsize=8, color="navy",
            bbox=dict(boxstyle="round,pad=0.3", edgecolor="navy", facecolor="white"))
plt.tight_layout(pad=1.5)
plt.savefig("mean_diff_heatmap.png")
plt.show()

# 6. Heatmap: Correlation
fig = plt.figure(figsize=(10, 7))
sns.heatmap(corr, annot=True, cmap="coolwarm", annot_kws={"size": 8, "weight": "bold"},
            cbar_kws={"label": "Correlation"})
plt.title("Correlation Heatmap", fontsize=16, weight="bold", pad=10)
plt.xticks(rotation=45, ha="right", fontsize=10)
plt.yticks(fontsize=10)
plt.gca().spines["top"].set_visible(False)
plt.gca().spines["right"].set_visible(False)
plt.annotate("Chart: Heatmap", xy=(0.02, 0.90), xycoords="axes fraction", fontsize=8, color="navy",
             bbox=dict(boxstyle="round,pad=0.3", edgecolor="navy", facecolor="white"))
plt.tight_layout(pad=1.5)
plt.savefig("corr_heatmap.png")
plt.show()

# 7. Scatter Plot
fig, ax = plt.subplots(figsize=(10, 7))
colors = {"Graduate": "#FF6B6B", "Dropout": "#40C4FF", "Enrolled": "#FFD700"}
for target in df["Target"].unique():
    subset = df[df["Target"] == target]
    sns.scatterplot(x="Admission grade", y="Curricular units 1st sem (approved)", color=colors[target],
                    label=target, data=subset, alpha=0.7, ax=ax, s=100)
    z = np.polyfit(subset["Admission grade"], subset["Curricular units 1st sem (approved)"], 1)
    p = np.poly1d(z)
    ax.plot(subset["Admission grade"], p(subset["Admission grade"]), color=colors[target], linestyle="--", linewidth=3)
ax.set_title("Admission Grade vs. 1st Sem Approved Units by Outcome", fontsize=16, weight="bold", pad=10)
ax.set_xlabel("Admission Grade", fontsize=14)
ax.set_ylabel("1st Sem Approved Units", fontsize=14)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(True, linestyle="--", alpha=0.7)
ax.legend(title="Outcome", fontsize=12, loc="upper left")
ax.annotate("Chart: Scatter Plot", xy=(0.02, 0.90), xycoords="axes fraction", fontsize=8, color="navy",
            bbox=dict(boxstyle="round,pad=0.3", edgecolor="navy", facecolor="white"))
plt.tight_layout(pad=1.5)
plt.savefig("scatter_plot.png")
plt.show()

# 8. Bar Plot
bar_data = df.groupby("Target")["Curricular units 1st sem (approved)"].agg(["mean", "sem"]).reset_index()
fig, ax = plt.subplots(figsize=(10, 7))
sns.barplot(x="Target", y="mean", hue="Target", palette=sns.light_palette("seagreen", n_colors=3), data=bar_data, ax=ax, legend=False)
ax.errorbar(x=bar_data["Target"], y=bar_data["mean"], yerr=bar_data["sem"], fmt="none", c="black", capsize=3)
ax.set_title("Mean 1st Sem Approved Units by Outcome", fontsize=16, weight="bold", pad=10)
ax.set_xlabel("Outcome", fontsize=14)
ax.set_ylabel("Mean Approved Units", fontsize=14)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
for i, row in enumerate(bar_data.itertuples()):
    ax.text(i, row.mean + 0.2, f"{row.mean:.1f}", ha="center", fontsize=8)
    if i == 0:  # Graduate vs. Dropout
        pct_change = ((bar_data["mean"][0] - bar_data["mean"][1]) / bar_data["mean"][1]) * 100
        ax.text(0.5, max(bar_data["mean"]) + 0.5, f"+{pct_change:.0f}%", ha="center", fontsize=8, color="darkred")
ax.annotate("Graduate lead!", xy=(0.05, 0.85), xycoords="axes fraction", fontsize=8, color="darkred",
            bbox=dict(boxstyle="round", facecolor="white"))
ax.annotate("Chart: Bar Plot", xy=(0.02, 0.90), xycoords="axes fraction", fontsize=8, color="navy",
            bbox=dict(boxstyle="round,pad=0.3", edgecolor="navy", facecolor="white"))
plt.tight_layout(pad=1.5)
plt.savefig("bar_plot.png")
plt.show()

# --- Z-test ---
print("\n=== Z-test for Admission Grade ===")
grad = df[df["Target"] == "Graduate"]["Admission grade"]
drop = df[df["Target"] == "Dropout"]["Admission grade"]
z_stat = (grad.mean() - drop.mean()) / np.sqrt((grad.std()**2 / len(grad)) + (drop.std()**2 / len(drop)))
p_val_z = 2 * (1 - norm.cdf(abs(z_stat)))
print(f"Z-test: z={z_stat:.2f}, p-value={p_val_z:.4f}")

# --- IQR ---
print("\n=== IQR ===")
for col in ["Admission grade"] + curricular_cols:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    print(f"{col}: IQR={q3 - q1:.2f}")

# --- Conclusion ---
print("\n=== Conclusion: Dropout and Academic Success ===")
print(f"Dropout Rate: {dropout_rate:.2f}%")
print(f"Success Rate (Graduate + Enrolled): {success_rate:.2f}%")
print("Key Insights:")
print(f"- Dropout rate ({dropout_rate:.2f}%) is significant, success rate is higher ({success_rate:.2f}%).")
print("- Graduates have higher admission grades (~128.8 vs. ~125.0), linked to lower dropout.")
print("- Scholarships reduce dropout (12.8% vs. 25.7%), boosting success.")
print("- Graduates approve more units (~6.4 vs. ~2.0), critical for academic success.")
print("- Moderate correlations (~0.3) show admission grades predict success weakly.")

# --- Predictive Model: Random Forest Classifier ---
print("\n=== Predictive Model for Dropout and Success ===")
# Select features based on EDA
features = [
    "Admission grade", "Scholarship holder",
    "Curricular units 1st sem (approved)", "Curricular units 1st sem (grade)",
    "Curricular units 2nd sem (approved)", "Curricular units 2nd sem (grade)"
]
X = df[features]
y = df["Target"]

# Encode target variable
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

# Train Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Evaluate model
y_pred = rf.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))  # Fixed: Compare y_test and y_pred
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Feature importance
feature_importance = pd.DataFrame({
    "Feature": [short_names.get(f, f) for f in features],
    "Importance": rf.feature_importances_
}).sort_values("Importance", ascending=False)
print("\nFeature Importance:")
print(feature_importance.round(3))

# Simulate future trends
print("\n=== Future Trends Prediction ===")
pred_counts = pd.Series(le.inverse_transform(y_pred)).value_counts(normalize=True) * 100
print("Predicted Outcome Percentages:")
print(pred_counts.round(2))
pred_dropout_rate = pred_counts.get("Dropout", 0)
pred_success_rate = pred_counts.get("Graduate", 0) + pred_counts.get("Enrolled", 0)
print(f"Predicted Dropout Rate: {pred_dropout_rate:.2f}%")
print(f"Predicted Success Rate: {pred_success_rate:.2f}%")

# Plot feature importance
fig, ax = plt.subplots(figsize=(10, 7))
sns.barplot(x="Importance", y="Feature", data=feature_importance, ax=ax)  # Removed palette to fix warning
ax.set_title("Feature Importance for Predicting Dropout/Success", fontsize=16, weight="bold")
ax.set_xlabel("Importance", fontsize=14)
ax.set_ylabel("Feature", fontsize=14)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.annotate("Chart: Feature Importance", xy=(0.02, 0.90), xycoords="axes fraction", fontsize=8, color="navy",
            bbox=dict(boxstyle="round,pad=0.3", edgecolor="navy", facecolor="white"))
plt.tight_layout(pad=1.5)
plt.savefig("feature_importance.png")
plt.show()