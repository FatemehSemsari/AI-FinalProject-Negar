from xgboost import XGBClassifier

def build_xgb_model(random_state=42):
    model = XGBClassifier(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.5,
        random_state=random_state,
        eval_metric="logloss",
        tree_method="hist",
        colsample_bylevel=1,
        min_child_weight=2,
    )
    return model