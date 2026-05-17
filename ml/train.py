# ml/train.py
from sklearn.metrics import accuracy_score # 성능지표저장을위해
import mlflow.sklearn # mlflow형태로저장
import os
import joblib
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, "data", "spam_short.csv")
ARTIFACT_DIR = os.path.join(BASE_DIR, "artifacts")
MODEL_PATH = os.path.join(ARTIFACT_DIR, "spam_model.joblib")
os.makedirs(ARTIFACT_DIR, exist_ok=True)

# 실험세팅
mlflow.set_tracking_uri("sqlite:///mlflow.db") # mlflow는sqlite를기본사용
mlflow.set_experiment("spam-classification-local")

df = pd.read_csv(DATA_PATH)
X = df["text"]
y = df["label"]
pipeline = Pipeline([
  ("vectorizer", CountVectorizer()),
  ("classifier", LogisticRegression(max_iter=200))
])

# 실험기록시작
with mlflow.start_run():
# 실험설정기록
    mlflow.log_param("model_type", "LogisticRegression")
    mlflow.log_param("vectorizer", "CountVectorizer")
    mlflow.log_param("max_iter", 200)
    mlflow.log_param("data_path", DATA_PATH)
    mlflow.log_param("row_count", len(df))
    
    pipeline.fit(X, y)

# 간단한metric 저장(train accuracy)
    preds = pipeline.predict(X)
    acc = accuracy_score(y, preds)
    mlflow.log_metric("train_accuracy", acc)
    
    
    joblib.dump(pipeline, MODEL_PATH)
    
    mlflow.log_artifact(DATA_PATH) # 데이터= 실무에서는데이터는따로관리
    mlflow.log_artifact(MODEL_PATH) # 모델파일

# MLflow모델형식으로도저장
    mlflow.sklearn.log_model(pipeline, name="model")
    
    print(f"Model saved to: {MODEL_PATH}")
    print(f"train_accuracy: {acc:.4f}")