import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)
from sklearn.preprocessing import LabelBinarizer
from xgboost import XGBRegressor, XGBClassifier

from utils.preprocessing import prepare_training_data

DATASET_PATH = os.path.join('dataset', 'vgsales.csv')
MODELS_DIR = 'models'


def train_regression_models(X_train, X_test, y_train, y_test, feature_names):
    results = {}
    models = {}

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    models['linear_regression'] = lr

    rf = RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    models['random_forest'] = rf

    xgb = XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05, random_state=42, n_jobs=-1)
    xgb.fit(X_train, y_train)
    models['xgboost'] = xgb

    for name, model in models.items():
        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)

        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_.tolist()
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_).tolist()
        else:
            importances = [0] * len(feature_names)

        results[name] = {
            'mae': float(mae),
            'rmse': float(rmse),
            'r2': float(r2),
            'predictions': preds.tolist(),
            'actual': y_test.tolist(),
            'feature_importance': dict(zip(feature_names, importances))
        }

    return models, results


def train_classification_models(X_train, X_test, y_train, y_test):
    results = {}
    models = {}

    classes = sorted(y_train.unique().tolist())

    rf = RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    models['random_forest'] = rf

    xgb_train = y_train.map({c: i for i, c in enumerate(classes)})
    xgb_test = y_test.map({c: i for i, c in enumerate(classes)})
    xgb = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05, random_state=42, n_jobs=-1, eval_metric='mlogloss')
    xgb.fit(X_train, xgb_train)
    models['xgboost'] = xgb

    for name, model in models.items():
        if name == 'xgboost':
            preds_encoded = model.predict(X_test)
            preds = pd.Series(preds_encoded).map({i: c for i, c in enumerate(classes)}).values
            probs = model.predict_proba(X_test)
        else:
            preds = model.predict(X_test)
            probs = model.predict_proba(X_test)

        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, average='weighted', zero_division=0)
        rec = recall_score(y_test, preds, average='weighted', zero_division=0)
        f1 = f1_score(y_test, preds, average='weighted', zero_division=0)
        cm = confusion_matrix(y_test, preds, labels=classes)
        report = classification_report(y_test, preds, labels=classes, zero_division=0, output_dict=True)

        lb = LabelBinarizer()
        y_test_bin = lb.fit_transform(y_test)
        roc_data = {}
        if y_test_bin.shape[1] > 1:
            for i, cls in enumerate(lb.classes_):
                fpr, tpr, _ = roc_curve(y_test_bin[:, i], probs[:, i])
                roc_data[cls] = {'fpr': fpr.tolist(), 'tpr': tpr.tolist(), 'auc': float(auc(fpr, tpr))}

        results[name] = {
            'accuracy': float(acc),
            'precision': float(prec),
            'recall': float(rec),
            'f1': float(f1),
            'confusion_matrix': cm.tolist(),
            'labels': classes,
            'classification_report': report,
            'roc': roc_data
        }

    return models, results


def run_training():
    os.makedirs(MODELS_DIR, exist_ok=True)

    data = prepare_training_data(DATASET_PATH)
    X = data['features']
    y_reg = data['labels_regression']
    y_clf = data['labels_classification']

    X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = train_test_split(
        X, y_reg, y_clf, test_size=0.2, random_state=42
    )

    reg_models, reg_results = train_regression_models(X_train, X_test, y_reg_train, y_reg_test, data['feature_columns'])
    clf_models, clf_results = train_classification_models(X_train, X_test, y_clf_train, y_clf_test)

    joblib.dump(reg_models['linear_regression'], os.path.join(MODELS_DIR, 'linear_regression.pkl'))
    joblib.dump(reg_models['random_forest'], os.path.join(MODELS_DIR, 'random_forest_regressor.pkl'))
    joblib.dump(reg_models['xgboost'], os.path.join(MODELS_DIR, 'xgboost_regressor.pkl'))

    joblib.dump(clf_models['random_forest'], os.path.join(MODELS_DIR, 'random_forest_classifier.pkl'))
    joblib.dump(clf_models['xgboost'], os.path.join(MODELS_DIR, 'xgboost_classifier.pkl'))

    joblib.dump(data['encoders'], os.path.join(MODELS_DIR, 'encoders.pkl'))
    joblib.dump(data['scaler'], os.path.join(MODELS_DIR, 'scaler.pkl'))
    joblib.dump(data['feature_columns'], os.path.join(MODELS_DIR, 'feature_columns.pkl'))
    joblib.dump(sorted(y_clf.unique().tolist()), os.path.join(MODELS_DIR, 'class_labels.pkl'))

    with open(os.path.join(MODELS_DIR, 'regression_results.json'), 'w') as f:
        json.dump(reg_results, f)

    with open(os.path.join(MODELS_DIR, 'classification_results.json'), 'w') as f:
        json.dump(clf_results, f)

    metadata = {
        'target_column': data['target'],
        'n_rows': int(data['raw_clean_df'].shape[0]),
        'n_features': int(X.shape[1]),
        'feature_columns': data['feature_columns'],
        'median_year': data['median_year']
    }
    with open(os.path.join(MODELS_DIR, 'metadata.json'), 'w') as f:
        json.dump(metadata, f)

    print('training complete')
    for name, res in reg_results.items():
        print(name, 'mae', round(res['mae'], 4), 'rmse', round(res['rmse'], 4), 'r2', round(res['r2'], 4))
    for name, res in clf_results.items():
        print(name, 'accuracy', round(res['accuracy'], 4), 'f1', round(res['f1'], 4))


if __name__ == '__main__':
    run_training()
