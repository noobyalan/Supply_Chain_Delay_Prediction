from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
from sqlalchemy import create_engine
def tune_hyperparameters(
        model, param_grid, X_train, y_train, X_test, y_test,
        scoring='f1', cv=5, verbose=1
    ):
    """
    使用 GridSearchCV 调参并在测试集上评估最佳模型

    参数:
        model       : sklearn 风格的模型实例，例如 XGBClassifier()
        param_grid  : dict，待搜索的超参数范围
        X_train     : 训练特征
        y_train     : 训练标签
        X_test      : 测试特征
        y_test      : 测试标签
        scoring     : 调参过程优化指标，默认'f1'
        cv          : 交叉验证折数
        verbose     : GridSearch 输出等级

    返回:
        best_model  : 调参后最佳模型（已拟合）
        best_params : 最佳超参数字典
        metrics     : 测试集评估结果字典
    """
    print("[INFO] 开始超参数搜索...")
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring=scoring,
        cv=cv,
        verbose=verbose,
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    print(f"[INFO] 最佳参数: {best_params}")
    print(f"[INFO] 最佳CV分数: {grid_search.best_score_:.4f}")

    # —— 测试集评估 ——
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1] if hasattr(best_model, "predict_proba") else None
    
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
    }
    if y_proba is not None:
        metrics["auc"] = roc_auc_score(y_test, y_proba)

    print("\n[INFO] 测试集评估结果：")
    for k, v in metrics.items():
        print(f"{k.capitalize():<9}: {v:.4f}")
    print("\n[INFO] 分类报告:\n", classification_report(y_test, y_pred))

    return best_model, best_params, metrics


    """ postgresql 连接"""
def get_postgres_engine(user, password, host, port, db):
    """生成 PostgreSQL Engine"""
    conn_str = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}"
    return create_engine(conn_str)