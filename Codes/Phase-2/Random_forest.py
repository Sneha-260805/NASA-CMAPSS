import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# ── Configuration ─────────────────────────────────────────────────────────────
RANDOM_STATE = 42
FILE_PATH    = 'clustered_01.csv','clustered_02.csv','clustered_03.csv','clustered_04','clustered_01_03.csv','clustered_02_04.csv','clustered_01_02_03_04.csv' # update if needed
TARGET_COL   = "cluster"
ID_COLS      = ["engine", "cycle"]  # columns to exclude from X

# ── Load ──────────────────────────────────────────────────────────────────────
df = pd.read_csv(FILE_PATH)

# ── 1) Check for missing targets ───────────────────────────────────────────────
n_missing = df[TARGET_COL].isna().sum()
print(f"Rows with missing '{TARGET_COL}': {n_missing}")

# ── 2) Drop any rows where the target is NaN ──────────────────────────────────
df = df.dropna(subset=[TARGET_COL])
df[TARGET_COL] = df[TARGET_COL].astype(int)  # ensure integer labels

# ── 3) Split into X / y ───────────────────────────────────────────────────────
X = df.drop(columns=ID_COLS + [TARGET_COL])
y = df[TARGET_COL]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    stratify=y,
    random_state=RANDOM_STATE
)

# ── 4) Train Random Forest ────────────────────────────────────────────────────
rf = RandomForestClassifier(
    n_estimators=500,
    class_weight="balanced",
    n_jobs=-1,
    random_state=RANDOM_STATE
)
rf.fit(X_train, y_train)

# ── 5) Evaluate ───────────────────────────────────────────────────────────────
y_pred = rf.predict(X_test)

print("\nRandom Forest classification report (test set):\n")
print(classification_report(y_test, y_pred, digits=3))

# ── 6) Confusion Matrix ────────────────────────────────────────────────────────
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=rf.classes_)
disp.plot(cmap="Blues", xticks_rotation=45)
plt.title("Random Forest Confusion Matrix 01")
plt.tight_layout()
plt.show()
