from pathlib import Path
from itertools import combinations
import json
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.inspection import permutation_importance

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR = BASE_DIR / "outputs"
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

input_file = DATA_DIR / "boston_boardie_analytics.csv"
df = pd.read_csv(input_file)

# 1. Data preparation for model and dashboard integration
for c in ["host_is_superhost", "instant_bookable", "neighbourhood_cleansed", "property_type", "room_type", "price_band"]:
    df[c] = df[c].astype(str).str.strip().replace({"nan": "Unknown"})

num_cols = ["accommodates", "bedrooms", "beds", "amenities_count", "availability_365", "number_of_reviews", "review_scores_rating", "availability_rate", "price"]
for c in num_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")
    df[c] = df[c].fillna(df[c].median())

df["availability_label"] = pd.cut(df["availability_rate"], bins=[-0.01, .25, .50, .75, 1.01], labels=["Low Availability", "Moderate Availability", "High Availability", "Very High Availability"])
df["review_quality"] = pd.cut(df["review_scores_rating"], bins=[0, 4.0, 4.5, 4.8, 5.01], labels=["Needs Review", "Good", "Very Good", "Excellent"])
df["amenity_level"] = pd.cut(df["amenities_count"], bins=[-1, 10, 20, 30, 999], labels=["Basic Amenities", "Standard Amenities", "High Amenities", "Premium Amenities"])
df["listing_size"] = pd.cut(df["accommodates"], bins=[0, 1, 2, 4, 999], labels=["Solo", "Pair", "Small Group", "Large Group"])
df.to_csv(DATA_DIR / "boston_boardie_analytics.csv", index=False)

# 2. Random Forest classification model
features = ["host_is_superhost", "neighbourhood_cleansed", "property_type", "room_type", "accommodates", "bedrooms", "beds", "amenities_count", "availability_365", "number_of_reviews", "review_scores_rating", "instant_bookable", "availability_rate"]
target = "price_band"
X = df[features]
y = df[target]
cat_features = ["host_is_superhost", "neighbourhood_cleansed", "property_type", "room_type", "instant_bookable"]
num_features = [c for c in features if c not in cat_features]

preprocessor = ColumnTransformer([
    ("categorical", OneHotEncoder(handle_unknown="ignore"), cat_features),
    ("numeric", StandardScaler(), num_features)
])
pipe = Pipeline([
    ("preprocess", preprocessor),
    ("model", RandomForestClassifier(random_state=42, class_weight="balanced"))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
param_grid = {
    "model__n_estimators": [100],
    "model__max_depth": [10, None],
    "model__min_samples_leaf": [1, 3]
}
grid = GridSearchCV(pipe, param_grid=param_grid, cv=3, scoring="f1_weighted", n_jobs=1)
grid.fit(X_train, y_train)
best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)

metrics = {
    "accuracy": accuracy_score(y_test, y_pred),
    "precision_weighted": precision_score(y_test, y_pred, average="weighted", zero_division=0),
    "recall_weighted": recall_score(y_test, y_pred, average="weighted", zero_division=0),
    "f1_weighted": f1_score(y_test, y_pred, average="weighted", zero_division=0),
    "best_parameters": grid.best_params_,
    "train_rows": len(X_train),
    "test_rows": len(X_test)
}

pd.DataFrame([
    {"Metric": "Accuracy", "Value": metrics["accuracy"]},
    {"Metric": "Weighted Precision", "Value": metrics["precision_weighted"]},
    {"Metric": "Weighted Recall", "Value": metrics["recall_weighted"]},
    {"Metric": "Weighted F1 Score", "Value": metrics["f1_weighted"]},
    {"Metric": "Training Rows", "Value": metrics["train_rows"]},
    {"Metric": "Testing Rows", "Value": metrics["test_rows"]},
    {"Metric": "Best Parameters", "Value": json.dumps(metrics["best_parameters"])}
]).to_csv(DATA_DIR / "model_metrics_summary.csv", index=False)

pd.DataFrame(classification_report(y_test, y_pred, zero_division=0, output_dict=True)).T.reset_index().rename(columns={"index": "Class_or_Average"}).to_csv(DATA_DIR / "classification_report.csv", index=False)

labels = sorted(y.unique().tolist())
cm = confusion_matrix(y_test, y_pred, labels=labels)
pd.DataFrame(cm, index=[f"Actual {x}" for x in labels], columns=[f"Predicted {x}" for x in labels]).to_csv(DATA_DIR / "confusion_matrix.csv")

perm = permutation_importance(best_model, X_test, y_test, n_repeats=4, random_state=42, scoring="f1_weighted", n_jobs=1)
importance = pd.DataFrame({"feature": features, "importance_mean": perm.importances_mean, "importance_std": perm.importances_std}).sort_values("importance_mean", ascending=False)
importance.to_csv(DATA_DIR / "feature_importance.csv", index=False)

# 3. External visualization tool exports
external_data = df.groupby(["neighbourhood_cleansed", "room_type", "price_band", "availability_label", "review_quality"], observed=True).agg(
    listings=("id", "count"),
    median_price=("price", "median"),
    avg_price=("price", "mean"),
    avg_availability=("availability_rate", "mean"),
    avg_review_score=("review_scores_rating", "mean"),
    avg_amenities=("amenities_count", "mean")
).reset_index()
external_data.to_csv(DATA_DIR / "tableau_boardie_dashboard_data.csv", index=False)

# 4. Association-pattern summary for Chapter 9 alignment
transactions = []
for _, r in df.iterrows():
    transactions.append(set(map(str, [
        f"room:{r['room_type']}",
        f"price_band:{r['price_band']}",
        f"availability:{r['availability_label']}",
        f"review:{r['review_quality']}",
        f"amenities:{r['amenity_level']}",
        f"size:{r['listing_size']}",
    ])))
N = len(transactions)
items = sorted(set().union(*transactions))
frequent = {}
for k in [1, 2, 3]:
    for combo in combinations(items, k):
        itemset = frozenset(combo)
        support = sum(1 for t in transactions if itemset.issubset(t)) / N
        if support >= 0.08:
            frequent[itemset] = support
rules = []
for itemset, sup_xy in frequent.items():
    if len(itemset) < 2:
        continue
    for r in range(1, len(itemset)):
        for left_tuple in combinations(itemset, r):
            left = frozenset(left_tuple)
            right = itemset - left
            if left in frequent and right in frequent:
                confidence = sup_xy / frequent[left]
                lift = confidence / frequent[right]
                if confidence >= 0.45 and lift >= 1.0:
                    rules.append({"antecedent": ", ".join(sorted(left)), "consequent": ", ".join(sorted(right)), "support": sup_xy, "confidence": confidence, "lift": lift})
rules_df = pd.DataFrame(rules).sort_values(["lift", "confidence", "support"], ascending=False).head(30)
rules_df.to_csv(DATA_DIR / "association_rules_summary.csv", index=False)

# 5. Professional visualizations
plt.rcParams.update({"font.size": 10})
fig, ax = plt.subplots(figsize=(7, 5))
im = ax.imshow(cm, interpolation="nearest")
ax.set_xticks(range(len(labels)))
ax.set_yticks(range(len(labels)))
ax.set_xticklabels(labels, rotation=35, ha="right")
ax.set_yticklabels(labels)
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, str(cm[i, j]), ha="center", va="center")
ax.set_title("Random Forest Confusion Matrix")
ax.set_xlabel("Predicted Price Band")
ax.set_ylabel("Actual Price Band")
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=180)
plt.close()

fig, ax = plt.subplots(figsize=(8, 5))
importance.head(10).sort_values("importance_mean").plot(kind="barh", x="feature", y="importance_mean", ax=ax, legend=False)
ax.set_title("Top Feature Importance for Price Band Classification")
ax.set_xlabel("Permutation Importance")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "feature_importance.png", dpi=180)
plt.close()

fig, ax = plt.subplots(figsize=(7, 5))
df["price_band"].value_counts().reindex(["Budget", "Mid-range", "Upper-mid", "Premium"]).plot(kind="bar", ax=ax)
ax.set_title("Listing Distribution by Price Band")
ax.set_xlabel("Price Band")
ax.set_ylabel("Number of Listings")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "price_band_distribution.png", dpi=180)
plt.close()

fig, ax = plt.subplots(figsize=(9, 5))
pd.crosstab(df["room_type"], df["price_band"]).plot(kind="bar", stacked=True, ax=ax)
ax.set_title("Price Band Composition by Room Type")
ax.set_xlabel("Room Type")
ax.set_ylabel("Number of Listings")
ax.legend(title="Price Band", bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "room_type_price_band.png", dpi=180)
plt.close()

fig, ax = plt.subplots(figsize=(7, 5))
df.groupby("price_band", observed=True)["availability_rate"].mean().reindex(["Budget", "Mid-range", "Upper-mid", "Premium"]).plot(kind="bar", ax=ax)
ax.set_title("Average Availability Rate by Price Band")
ax.set_xlabel("Price Band")
ax.set_ylabel("Average Availability Rate")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "availability_by_price_band.png", dpi=180)
plt.close()

fig, ax = plt.subplots(figsize=(10, 6))
df.groupby("neighbourhood_cleansed")["price"].median().sort_values(ascending=False).head(12).sort_values().plot(kind="barh", ax=ax)
ax.set_title("Top 12 Areas by Median Listing Price")
ax.set_xlabel("Median Price")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "neighbourhood_median_price.png", dpi=180)
plt.close()

if not rules_df.empty:
    fig, ax = plt.subplots(figsize=(10, 6))
    rules_plot = rules_df.head(8).copy()
    rule_labels = rules_plot["antecedent"].str.replace("price_band:", "", regex=False).str.replace("room:", "", regex=False).str[:30] + " -> " + rules_plot["consequent"].str.replace("price_band:", "", regex=False).str.replace("room:", "", regex=False).str[:30]
    ax.barh(rule_labels.iloc[::-1], rules_plot["lift"].iloc[::-1])
    ax.set_title("Top Association Rules by Lift")
    ax.set_xlabel("Lift")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "association_rules_lift.png", dpi=180)
    plt.close()

# 6. Dashboard preview
chart_files = ["price_band_distribution.png", "room_type_price_band.png", "availability_by_price_band.png", "feature_importance.png", "confusion_matrix.png", "neighbourhood_median_price.png", "association_rules_lift.png"]
html = f"""<!DOCTYPE html><html><head><meta charset='utf-8'><title>Boardie Task 9 Dashboard Preview</title>
<style>body{{font-family:Arial,sans-serif;margin:30px;background:#f7f7f7;color:#222}}.card{{background:white;border-radius:14px;padding:18px;margin:18px 0;box-shadow:0 2px 10px #ddd}}img{{max-width:100%;border:1px solid #ddd;border-radius:8px}}.metrics{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}}.metric{{background:#fff;border-radius:12px;padding:18px;text-align:center;box-shadow:0 2px 10px #ddd}}.metric b{{font-size:26px;display:block;margin-top:8px}}</style></head><body>
<h1>Boardie Analytics Task 9 Dashboard Preview</h1><p>This preview shows the professional visualization layer of the integrated analytics pipeline.</p>
<div class='metrics'><div class='metric'>Accuracy<b>{metrics['accuracy']:.3f}</b></div><div class='metric'>Weighted Precision<b>{metrics['precision_weighted']:.3f}</b></div><div class='metric'>Weighted Recall<b>{metrics['recall_weighted']:.3f}</b></div><div class='metric'>Weighted F1<b>{metrics['f1_weighted']:.3f}</b></div></div>
"""
for chart in chart_files:
    if (OUTPUT_DIR / chart).exists():
        html += f"<div class='card'><h2>{chart.replace('_', ' ').replace('.png', '').title()}</h2><img src='{chart}'></div>\n"
html += "</body></html>"
(OUTPUT_DIR / "dashboard_preview.html").write_text(html, encoding="utf-8")

results_text = f"""Boardie Analytics Task 9 Model Results

Dataset rows: {len(df)}
Training rows: {len(X_train)}
Testing rows: {len(X_test)}
Target variable: price_band
Model: Random Forest Classifier with preprocessing and GridSearchCV fine-tuning
Best parameters: {json.dumps(metrics['best_parameters'])}

Algorithm Metrics:
Accuracy: {metrics['accuracy']:.4f}
Weighted Precision: {metrics['precision_weighted']:.4f}
Weighted Recall: {metrics['recall_weighted']:.4f}
Weighted F1 Score: {metrics['f1_weighted']:.4f}

External Tool Integration Outputs:
- tableau_boardie_dashboard_data.csv
- model_metrics_summary.csv
- feature_importance.csv
- classification_report.csv
- association_rules_summary.csv
- dashboard_preview.html
"""
(OUTPUT_DIR / "model_results.txt").write_text(results_text, encoding="utf-8")
print("Task 9 pipeline completed. Check data/processed and outputs folders.")
