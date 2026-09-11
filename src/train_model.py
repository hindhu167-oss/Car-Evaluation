from pathlib import Path
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "car_evaluation.csv"
MODEL = ROOT / "models" / "car_evaluation_model.pkl"

df = pd.read_csv(DATA)

X = df.drop("evaluation", axis=1)
y = df["evaluation"]

preprocessor = ColumnTransformer([
    ("categorical", OneHotEncoder(handle_unknown="ignore"), X.columns)
])

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", DecisionTreeClassifier(random_state=42, max_depth=8))
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_test)

print("Dataset shape:", df.shape)
print("Accuracy:", accuracy_score(y_test, predictions))
print("\nClassification Report:")
print(classification_report(y_test, predictions, zero_division=0))

MODEL.parent.mkdir(exist_ok=True)
joblib.dump(pipeline, MODEL)

print("\nModel saved successfully:", MODEL)
