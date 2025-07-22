import pandas as pd
import plotly.express as px
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

def train_model(features_train, input_features, target):
    X = features_train[input_features]
    y = features_train[target]
    model = LogisticRegression(max_iter=300)
    model.fit(X, y)
    return model

def evaluate_model(model, features_test, input_features, target):
    X = features_test[input_features]
    y_true = features_test[target]
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]
    report = classification_report(y_true, y_pred, output_dict=True)
    auc = roc_auc_score(y_true, y_proba)
    matrix = confusion_matrix(y_true, y_pred).tolist()

    return {
        "roc_auc": auc,
        "accuracy": report.get("accuracy"),
        "precision": report.get("weighted avg", {}).get("precision"),
        "recall": report.get("weighted avg", {}).get("recall"),
        "f1_score": report.get("weighted avg", {}).get("f1-score"),
        "confusion_matrix": matrix
    }

def feature_importance_plot(model, feature_names):
    if hasattr(model, "coef_"):
        coefs = model.coef_.flatten()
        df = pd.DataFrame({'feature': feature_names, 'importance': coefs})
        df = df.sort_values(by='importance')
        fig = px.bar(
            df,
            x='importance',
            y='feature',
            orientation='h',
            title='Feature Importance (Logistic Regression Coefficients)'
        )
        fig.update_layout(xaxis_title='Coefficient Value', yaxis_title='Feature')
        return fig
    else:
        return None
