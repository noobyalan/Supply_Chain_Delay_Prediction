# model.py
import xgboost as xgb
from sklearn.metrics import classification_report, roc_auc_score

class DelayPredictor:
    def __init__(self, **params):
        self.model = xgb.XGBClassifier(**params)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        preds = self.model.predict(X_test)
        proba = self.model.predict_proba(X_test)[:, 1]
        print(classification_report(y_test, preds))
        print("AUC:", roc_auc_score(y_test, proba))

    def save_model(self, path):
        self.model.save_model(path)

    def load_model(self, path):
        self.model = xgb.XGBClassifier()
        self.model.load_model(path)