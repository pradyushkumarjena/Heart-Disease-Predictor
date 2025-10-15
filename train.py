# ================= HEART DISEASE PREDICTION WORKFLOW =================

# 1. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc
)

# ================= 2. Load Dataset =================
df = pd.read_csv("data/heart.csv")

# ================= 3. Ensure 'target' Column Exists =================
if 'target' not in df.columns:
    for col in df.columns:
        if col.lower() in ['condition', 'output', 'diagnosis', 'heartdisease', 'target']:
            df.rename(columns={col: 'target'}, inplace=True)
            break

# ================= 4. Verify Target Values =================
print("Target value counts:\n", df['target'].value_counts(), "\n")
if df['target'].nunique() < 2:
    raise ValueError("Dataset has only one target class. Please verify your heart.csv file.")

# ================= 5. Data Overview =================
print("Dataset shape:", df.shape)
print(df.head())

# ================= 6. Replace 0s in Numeric Continuous Columns =================
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
for col in numeric_cols:
    if col != 'target' and (df[col] == 0).any():
        mean_value = df[col].replace(0, np.nan).mean()
        df[col] = df[col].replace(0, mean_value)

# ================= 7. Features and Target =================
X = df.drop('target', axis=1)
y = df['target']

# ================= 8. Train/Test Split =================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ================= 9. Feature Scaling =================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ================= 10. Define Models =================
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=200, random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=5)
}

# ================= 11. Train, Evaluate, and Store Results =================
results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Metrics
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    prec = precision_score(y_test, y_test_pred)
    rec = recall_score(y_test, y_test_pred)
    f1 = f1_score(y_test, y_test_pred)
    
    # Store results
    results.append([name, train_acc*100, test_acc*100, prec, rec, f1])
    
    # Print performance summary
    print(f"\n=== {name} ===")
    print(f"Training Accuracy: {train_acc*100:.2f}% | Testing Accuracy: {test_acc*100:.2f}%")
    print(f"Precision: {prec:.2f} | Recall: {rec:.2f} | F1 Score: {f1:.2f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_test_pred)
    print(f"Confusion Matrix for {name}:\n{cm}\n")

# ================= 12. Store All Results in a DataFrame =================
results_df = pd.DataFrame(
    results,
    columns=['Model', 'Train Accuracy', 'Test Accuracy', 'Precision', 'Recall', 'F1 Score']
)
print("\nPerformance Comparison Table:\n")
print(results_df)

# ================= 13. EXPLORATORY DATA ANALYSIS (EDA) =================
plt.figure(figsize=(6, 4))
sns.histplot(data=df, x='age', hue='target', bins=15, kde=True, palette='Set2')
plt.title("Heart Attack Risk Based on Age (Figure 1)")
plt.xlabel("Age")
plt.ylabel("Count")
plt.show()

plt.figure(figsize=(6, 4))
sns.histplot(data=df, x='trestbps', hue='target', bins=15, kde=True, palette='coolwarm')
plt.title("Heart Attack Risk Based on Resting Blood Pressure (Figure 2)")
plt.xlabel("Resting BP")
plt.ylabel("Count")
plt.show()

plt.figure(figsize=(5, 4))
sns.countplot(data=df, x='sex', hue='target', palette='Set1')
plt.title("Heart Disease Cases by Sex (Figure 3)")
plt.xlabel("Sex (0 = Female, 1 = Male)")
plt.ylabel("Count")
plt.show()

plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='cp', hue='target', palette='viridis')
plt.title("Heart Disease Cases by Chest Pain Type (Figure 4)")
plt.xlabel("Chest Pain Type (0–3)")
plt.ylabel("Count")
plt.show()

# ================= 14. MODEL TEST ACCURACY BAR CHART (Y-axis from 0) =================
plt.figure(figsize=(8, 5))
sns.barplot(
    data=results_df,
    x='Model',
    y='Test Accuracy',
    hue='Model',          # ✅ Added to avoid deprecation warning
    dodge=False,
    palette='pastel',
    edgecolor='black',
    legend=False
)

# Add accuracy labels
for i, v in enumerate(results_df['Test Accuracy']):
    plt.text(i, v + 0.8, f"{v:.2f}%", ha='center', fontsize=10, color='black')

plt.title("Model Test Accuracy Comparison (Figure 8)", fontsize=13)
plt.xlabel("Machine Learning Algorithms", fontsize=11)
plt.ylabel("Accuracy (%)", fontsize=11)
plt.ylim(0, 100)  # ✅ Y-axis starts from 0
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# ================= 15. BAR + LINE GRAPH COMBINATION =================
plt.figure(figsize=(8, 5))
x = np.arange(len(results_df['Model']))
accuracies = results_df['Test Accuracy']

# Bar plot (no deprecation warning)
sns.barplot(
    data=results_df,
    x='Model',
    y='Test Accuracy',
    hue='Model',
    dodge=False,
    palette='pastel',
    edgecolor='black',
    legend=False
)

# Line plot
plt.plot(x, accuracies, color='black', marker='o', linewidth=2, label='Accuracy Trend')

# Add labels
for i, v in enumerate(accuracies):
    plt.text(i, v + 0.8, f"{v:.2f}%", ha='center', fontsize=10)

plt.ylim(0, 100)
plt.title("Accuracy Comparison of Models (Bar + Line Graph)", fontsize=13)
plt.xlabel("Algorithms")
plt.ylabel("Accuracy (%)")
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# ================= 16. ROC–AUC CURVE FOR ALL MODELS =================
plt.figure(figsize=(8, 6))

for name, model in models.items():
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
    else:
        y_score = model.decision_function(X_test)
    
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {roc_auc:.2f})")

# Random chance line
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')

plt.title("ROC–AUC Curve Comparison (Figure 9)", fontsize=13)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
