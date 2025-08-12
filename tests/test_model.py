import os
import joblib
from sklearn.model_selection import train_test_split
from src.data_loader import load_raw_data
from src.feature_engineering import process_features
from src.model import DelayPredictor

def test_model_training_and_saving():
    df = load_raw_data()
    X, y, encoders = process_features(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = DelayPredictor(n_estimators=10, max_depth=3, learning_rate=0.1)
    model.train(X_train, y_train)
    model.evaluate(X_test, y_test)

    # 保存
    model_path = "models/test_model.json"
    enc_path = "models/test_encoders.pkl"
    model.save_model(model_path)
    joblib.dump(encoders, enc_path)

    assert os.path.exists(model_path)
    assert os.path.exists(enc_path)