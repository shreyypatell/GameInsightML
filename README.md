# GameInsight ML

Video Game Sales Prediction and Classification, built on the vgsales dataset.

## What it does

Predicts a video game's Global_Sales (regression) and its sales tier — Low, Medium, High (classification) — using only Platform, Genre, Publisher and Year. Regional sales columns are intentionally excluded from the features since they sum almost exactly to Global_Sales and would make the task trivial.

Models trained:

- Linear Regression, Random Forest Regressor, XGBoost Regressor
- Random Forest Classifier, XGBoost Classifier

## Local Setup

1. Install Python 3.11.

2. Create a virtual environment.

```
python -m venv venv
venv\Scripts\activate      (Windows)
source venv/bin/activate   (Mac/Linux)
```

3. Install dependencies.

```
pip install -r requirements.txt
```

4. Run the app.

```
python app.py
```

The first run trains all five models automatically and saves them to `models/`. Every run after that loads the saved models instead of retraining. Use the Retrain Models button on the About page if `dataset/vgsales.csv` changes.

5. Open `http://localhost:5000` in your browser.

## Project Structure

```
GameInsightML/
├── app.py
├── train_models.py
├── requirements.txt
├── runtime.txt
├── Procfile
├── vercel.json
├── netlify.toml
├── .gitignore
├── dataset/vgsales.csv
├── models/
├── uploads/
├── predictions/
├── notebooks/
├── utils/
│   ├── preprocessing.py
│   └── eda.py
├── templates/
└── static/
    ├── css/
    ├── js/
    └── images/
```

## Free Deployment

This is a Flask app with server-side model training and file uploads, so it needs a real Python process running continuously — Netlify cannot host that on its own, since Netlify only serves static files and short-lived serverless functions. The practical free setup is:

**Backend (Flask + models): Render**

1. Push this project to GitHub.
2. Create a free Web Service on Render, connect the repo.
3. Build command: `pip install -r requirements.txt`
4. Start command: `gunicorn app:app`
5. Render reads `Procfile` and `runtime.txt` automatically.
6. Wait for the first deploy to finish training the models (takes a minute or two).

Koyeb or Railway work the same way if Render's free tier isn't available in your region — both also read the `Procfile`.

**Frontend: same Flask app already serves the templates**, so a separate frontend host isn't required for grading purposes. If you specifically want the static assets served from Netlify/Vercel and only the API on Render, use the included `netlify.toml`, which proxies `/api/*` to the Render backend URL — replace `your-backend-url.onrender.com` with your actual Render URL after deploying.

`vercel.json` is included if you'd rather deploy the whole Flask app to Vercel instead of Render. Note that Vercel's serverless functions are stateless and have a read-only filesystem outside `/tmp`, so model retraining and CSV uploads work less reliably there than on Render — Render is the recommended option for this project.

## Retraining

Models are trained once and saved with joblib to `models/`. The app loads these saved files on startup instead of retraining every time. Use the Retrain Models button on the About page, or run `python train_models.py` directly, if the dataset changes.
