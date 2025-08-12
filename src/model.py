# src/model.py
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def create_pipeline(numeric_features, categorical_features, **model_params):
    """构造预处理 + 模型的训练Pipeline"""

    # 数值特征预处理
    numeric_transformer = Pipeline(steps=[
        ("scaler", StandardScaler())
    ])

    # 类别特征预处理
    categorical_transformer = Pipeline(steps=[
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    # 列组合
    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ])

    # 模型（这里用 XGBClassifier，将来换模型直接改这一行）
    clf = XGBClassifier(
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        **model_params
    )

    # 拼装 Pipeline
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", clf)
    ])

    return pipeline