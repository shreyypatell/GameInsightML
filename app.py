import os
import json
import joblib
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify, send_file, send_from_directory
from werkzeug.utils import secure_filename

from utils.preprocessing import load_raw_dataset, clean_dataset, prepare_inference_frame, normalize_uploaded_dataframe
from utils import eda
import train_models

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['PREDICTIONS_FOLDER'] = 'predictions'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

MODELS_DIR = 'models'
DATASET_PATH = os.path.join('dataset', 'vgsales.csv')

_state = {}


def load_state():
    global _state
    raw = load_raw_dataset(DATASET_PATH)
    clean_df, target = clean_dataset(raw)

    encoders = joblib.load(os.path.join(MODELS_DIR, 'encoders.pkl'))
    scaler = joblib.load(os.path.join(MODELS_DIR, 'scaler.pkl'))
    feature_columns = joblib.load(os.path.join(MODELS_DIR, 'feature_columns.pkl'))
    class_labels = joblib.load(os.path.join(MODELS_DIR, 'class_labels.pkl'))

    reg_models = {
        'linear_regression': joblib.load(os.path.join(MODELS_DIR, 'linear_regression.pkl')),
        'random_forest': joblib.load(os.path.join(MODELS_DIR, 'random_forest_regressor.pkl')),
        'xgboost': joblib.load(os.path.join(MODELS_DIR, 'xgboost_regressor.pkl'))
    }
    clf_models = {
        'random_forest': joblib.load(os.path.join(MODELS_DIR, 'random_forest_classifier.pkl')),
        'xgboost': joblib.load(os.path.join(MODELS_DIR, 'xgboost_classifier.pkl'))
    }

    with open(os.path.join(MODELS_DIR, 'regression_results.json')) as f:
        reg_results = json.load(f)
    with open(os.path.join(MODELS_DIR, 'classification_results.json')) as f:
        clf_results = json.load(f)
    with open(os.path.join(MODELS_DIR, 'metadata.json')) as f:
        metadata = json.load(f)

    _state = {
        'raw_df': raw,
        'clean_df': clean_df,
        'target': target,
        'encoders': encoders,
        'scaler': scaler,
        'feature_columns': feature_columns,
        'class_labels': class_labels,
        'reg_models': reg_models,
        'clf_models': clf_models,
        'reg_results': reg_results,
        'clf_results': clf_results,
        'metadata': metadata,
        'median_year': metadata.get('median_year', int(clean_df['Year'].median()))
    }


if not os.path.exists(os.path.join(MODELS_DIR, 'metadata.json')):
    train_models.run_training()
load_state()


@app.route('/')
def home():
    stats = {
        'total_games': int(_state['clean_df'].shape[0]),
        'total_platforms': int(_state['clean_df']['Platform'].nunique()),
        'total_genres': int(_state['clean_df']['Genre'].nunique()),
        'total_publishers': int(_state['clean_df']['Publisher'].nunique()),
        'best_r2': round(max(r['r2'] for r in _state['reg_results'].values()), 3),
        'best_accuracy': round(max(r['accuracy'] for r in _state['clf_results'].values()), 3)
    }
    return render_template('home.html', stats=stats)


@app.route('/dataset')
def dataset_page():
    overview = eda.dataset_overview(_state['clean_df'])
    sample_rows = _state['raw_df'].head(15).to_dict(orient='records')
    columns = _state['raw_df'].columns.tolist()
    return render_template('dataset.html', overview=overview, sample_rows=sample_rows, columns=columns)


@app.route('/eda')
def eda_page():
    return render_template('eda.html')


@app.route('/api/eda-charts')
def api_eda_charts():
    df = _state['clean_df']
    target = _state['target']
    charts = {
        'genre': eda.genre_distribution(df),
        'platform': eda.platform_distribution(df),
        'publisher': eda.publisher_distribution(df),
        'yearwise': eda.yearwise_releases(df),
        'sales_dist': eda.global_sales_distribution(df, target),
        'top_publishers': eda.top_publishers_by_sales(df, target),
        'correlation': eda.correlation_heatmap(df),
        'boxplot': eda.boxplot_by_genre(df, target),
        'histogram': eda.histogram_numeric(df, target)
    }
    top_games = eda.top_selling_games(_state['raw_df'], target)
    if top_games:
        charts['top_games'] = top_games
    pair = eda.pairplot_subset(df, target)
    if pair:
        charts['pairplot'] = pair
    return jsonify(charts)


@app.route('/regression')
def regression_page():
    return render_template('regression.html', results=_state['reg_results'])


@app.route('/classification')
def classification_page():
    return render_template('classification.html', results=_state['clf_results'])


@app.route('/comparison')
def comparison_page():
    return render_template('comparison.html', reg_results=_state['reg_results'], clf_results=_state['clf_results'])


@app.route('/manual-predict')
def manual_predict_page():
    df = _state['raw_df']
    platforms = sorted(df['Platform'].dropna().unique().tolist())
    genres = sorted(df['Genre'].dropna().unique().tolist())
    publishers = sorted(df['Publisher'].dropna().unique().tolist())
    return render_template('manual_predict.html', platforms=platforms, genres=genres, publishers=publishers)


@app.route('/api/manual-predict', methods=['POST'])
def api_manual_predict():
    payload = request.get_json()
    model_choice = payload.get('model', 'random_forest')

    input_row = {
        'Platform': payload.get('platform'),
        'Genre': payload.get('genre'),
        'Publisher': payload.get('publisher'),
        'Year': payload.get('year')
    }
    input_df = pd.DataFrame([input_row])

    features = prepare_inference_frame(
        input_df, _state['encoders'], _state['scaler'], _state['feature_columns']
    )

    reg_model = _state['reg_models'].get(model_choice, _state['reg_models']['random_forest'])
    predicted_sales = float(reg_model.predict(features)[0])
    predicted_sales = max(predicted_sales, 0)

    clf_model_choice = model_choice if model_choice in _state['clf_models'] else 'random_forest'
    clf_model = _state['clf_models'][clf_model_choice]

    if clf_model_choice == 'xgboost':
        pred_encoded = clf_model.predict(features)[0]
        predicted_category = _state['class_labels'][int(pred_encoded)]
    else:
        predicted_category = clf_model.predict(features)[0]

    return jsonify({
        'predicted_sales': round(predicted_sales, 3),
        'predicted_category': predicted_category
    })


@app.route('/upload-predict')
def upload_predict_page():
    return render_template('upload_predict.html')


@app.route('/api/upload-predict', methods=['POST'])
def api_upload_predict():
    if 'file' not in request.files:
        return jsonify({'error': 'no file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'empty filename'}), 400

    filename = secure_filename(file.filename)
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(save_path)

    model_choice = request.form.get('model', 'random_forest')

    try:
        upload_df = pd.read_csv(save_path)
    except Exception:
        return jsonify({'error': 'could not read csv file'}), 400

    if upload_df.shape[0] == 0:
        return jsonify({'error': 'the uploaded file has no rows'}), 400

    defaults = {
        'Platform': 'Unknown',
        'Genre': 'Unknown',
        'Publisher': 'Unknown',
        'Year': _state['median_year']
    }
    upload_df, column_mapping, defaulted_columns = normalize_uploaded_dataframe(upload_df, defaults)

    features = prepare_inference_frame(
        upload_df, _state['encoders'], _state['scaler'], _state['feature_columns']
    )

    reg_model = _state['reg_models'].get(model_choice, _state['reg_models']['random_forest'])
    predictions = reg_model.predict(features)
    predictions = np.clip(predictions, 0, None)

    clf_model_choice = model_choice if model_choice in _state['clf_models'] else 'random_forest'
    clf_model = _state['clf_models'][clf_model_choice]
    if clf_model_choice == 'xgboost':
        pred_encoded = clf_model.predict(features)
        categories = [_state['class_labels'][int(i)] for i in pred_encoded]
    else:
        categories = clf_model.predict(features).tolist()

    result_df = upload_df.copy()
    result_df['Predicted_Global_Sales'] = np.round(predictions, 3)
    result_df['Predicted_Category'] = categories

    output_filename = f'predictions_{filename}'
    output_path = os.path.join(app.config['PREDICTIONS_FOLDER'], output_filename)
    result_df.to_csv(output_path, index=False)

    preview_df = result_df.head(500).copy()
    preview_df = preview_df.replace([np.inf, -np.inf], np.nan)
    preview_df = preview_df.astype(object).where(pd.notnull(preview_df), None)

    return jsonify({
        'rows': preview_df.to_dict(orient='records'),
        'total_rows': int(result_df.shape[0]),
        'previewed_rows': int(preview_df.shape[0]),
        'download_url': f'/download/predictions/{output_filename}',
        'column_mapping': column_mapping,
        'defaulted_columns': defaulted_columns
    })


@app.route('/download/predictions/<filename>')
def download_prediction(filename):
    return send_from_directory(app.config['PREDICTIONS_FOLDER'], filename, as_attachment=True)


@app.route('/download/dataset')
def download_dataset():
    return send_from_directory('dataset', 'vgsales.csv', as_attachment=True)


@app.route('/download/model/<model_name>')
def download_model(model_name):
    filename_map = {
        'linear_regression': 'linear_regression.pkl',
        'random_forest_regressor': 'random_forest_regressor.pkl',
        'xgboost_regressor': 'xgboost_regressor.pkl',
        'random_forest_classifier': 'random_forest_classifier.pkl',
        'xgboost_classifier': 'xgboost_classifier.pkl'
    }
    if model_name not in filename_map:
        return jsonify({'error': 'unknown model'}), 404
    return send_from_directory(MODELS_DIR, filename_map[model_name], as_attachment=True)


@app.route('/api/retrain', methods=['POST'])
def api_retrain():
    train_models.run_training()
    load_state()
    return jsonify({'status': 'success', 'message': 'models retrained successfully'})


@app.route('/about')
def about_page():
    return render_template('about.html')


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
