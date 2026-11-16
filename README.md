# 🍿 Netflix Content Type Predictor System

An end-to-end machine learning project that predicts whether a Netflix title is a **Movie** or a **TV Show**.
It covers the whole path: data analysis in a notebook, a training script, a FastAPI prediction API, a Streamlit web app, and deployment on AWS EC2.

## What it does

You enter five details about a title and the model returns the predicted type with a probability.

| Input | Example | Notes |
|---|---|---|
| Director | `Not Given` | Use `Not Given` when the title has no director. The model only uses whether a director exists. |
| Date Added | `7/31/2020` | Format `M/D/YYYY`. Only the year is used. |
| Release Year | `2020` | Whole number. |
| Rating | `TV-PG` | One of: G, NC-17, NR, PG, PG-13, R, TV-14, TV-G, TV-MA, TV-PG, TV-Y, TV-Y7, TV-Y7-FV, UR. |
| Listed in | `British TV Shows, Crime TV Shows` | One or more Netflix genres, separated by commas. |

`duration` and `country` are **not used** on purpose. Duration gives the answer away (`1 Season` is always a TV Show, `110 min` is always a Movie), and many country labels are unreliable.

## Data and model

- **Dataset:** `Dataset/Dataset.csv`, 8,790 Netflix titles. 4 duplicates are removed, leaving 8,786 (6,123 Movies and 2,663 TV Shows, so about 30% are TV Shows).
- **Features (44):** 3 numbers (`release_year`, `year_added`, `has_director`), 14 one-hot rating columns and 27 one-hot genre columns. Genre names are cleaned first (the words TV, Movies, Shows, Series and Features are removed, and a few duplicate names are merged, for example `Docuseries` into `Documentaries`).
- **Model:** `StandardScaler` followed by an RBF Support Vector Classifier (`C=10`, `gamma=0.01`, `probability=True`). Seven classifiers were compared in the notebook, and Logistic Regression and SVC gave the best results. The SVC was tuned and kept.
- **Split:** 85% training and 15% test, stratified, `random_state=42` (1,318 test titles).

### Results (test set)

| Metric | Score |
|---|---|
| Accuracy | 97.57% |
| F1 (TV Show) | 0.960 |
| Recall (TV Show) | 95.74% |
| Precision (TV Show) | 96.22% |
| Mistakes | 32 of 1,318 |

Training accuracy is 98.31%, so the model is not badly overfitted.

## Project structure

```text
Netflix_Content_Type_Predictor_System/
├── Backend/
│   ├── main.py                 # FastAPI app and endpoints
│   ├── predictor.py            # loads the model and makes one prediction
│   └── training.py             # cleans the data, trains the model, saves it
├── frontend/
│   └── app.py                  # Streamlit web app (calls the API)
├── Dataset/
│   └── Dataset.csv
├── model_dir/
│   └── netflix-content-type-prediction-model.joblib   # saved model
├── notebook_files/
│   └── Netflix_content_type_prediction.ipynb          # analysis and model selection
├── logs/
│   └── app.log                 # created automatically (not in git)
├── env_template.txt            # copy to .env and fill in
├── EC2_API_streamlit_commands.txt
├── requirements.txt
└── README.md
```

The notebook follows these steps: 1. Setup, 2. Loading the dataset, 3. Exploratory data analysis, 4. Pre-processing, 5. Baseline model, 6. Model selection and hyperparameter tuning, 7. Retraining with the best parameters, 8. Building a predictive system.

## Tech stack

Python 3.14, pandas, NumPy, scikit-learn, FastAPI, Uvicorn, Streamlit, Pydantic, python-dotenv, joblib, matplotlib and seaborn (all versions are pinned in `requirements.txt`).

## Setup

Run every command from the project folder.

**1. Create a virtual environment and install the packages**

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows
# source .venv/bin/activate       # Linux / macOS
pip install -r requirements.txt
```

**2. Create the `.env` file**

Copy `env_template.txt` to `.env` and fill in the values. The `.env` file is in `.gitignore`, so it is never committed.

| Variable | Value |
|---|---|
| `PROJECT_ROOT` | Absolute path of this project folder |
| `LOG_DIR`, `DATASET_DIR`, `MODEL_DIR` | `logs`, `Dataset`, `model_dir` |
| `DATASET_NAME`, `LOG_NAME`, `MODEL_NAME` | `Dataset.csv`, `app.log`, `netflix-content-type-prediction-model.joblib` |
| `TARGET_COL` | `type` |
| `TEST_SIZE`, `RANDOM_STATE` | `0.15`, `42` |
| `API_URL` | `http://127.0.0.1:8000/netflix-type-prediction-system` |

**3. Train the model** (optional, the trained model is already in `model_dir/`)

```bash
python Backend/training.py
```

This prints the data summary, writes the metrics to `logs/app.log` and saves the model to `model_dir/`.

**4. Start the API**

```bash
uvicorn Backend.main:app --reload --port 8000
```

Interactive docs: http://127.0.0.1:8000/docs

**5. Start the web app** (in a second terminal)

```bash
streamlit run frontend/app.py
```

Open http://localhost:8501, fill in the details and click **Predict**.

## API

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/type` | Health check. Returns `{"status": "ok"}`. |
| POST | `/netflix-type-prediction-system` | Predicts the content type. |

**Request**

```json
{
  "director": "Not Given",
  "date_added": "7/31/2020",
  "release_year": 2020,
  "rating": "TV-PG",
  "listed_in": "British TV Shows, Crime TV Shows, International TV Shows"
}
```

**Response**

```json
{
  "prediction": 1,
  "probability": 0.9999999910598766,
  "content": "TV Show 📺"
}
```

- `prediction`: `0` is a Movie, `1` is a TV Show.
- `probability`: the model's chance (0 to 1) that the title is a **TV Show**. A Movie therefore has a low value, for example 0.03.
- `content`: a readable label (`Movie 🎥` or `TV Show 📺`).

**Things to know when calling the API directly**

- Send `"Not Given"` when there is no director. An empty string `""` is counted as having a director.
- `date_added` must be `M/D/YYYY`. Another format returns a 500 error. Missing fields return a 422 error.
- A rating that is not in the list above is accepted but ignored, so the answer may be unreliable.

## Deploy on AWS EC2

The commands are also in `EC2_API_streamlit_commands.txt`. On the server, clone the repo, create the virtual environment, install `requirements.txt` and create the `.env` file (set `PROJECT_ROOT` to the folder on the server). Then:

```bash
# start the API in the background
nohup uvicorn Backend.main:app --host 0.0.0.0 --port 8000 > uvicorn.log 2>&1 &
tail -f uvicorn.log

# start the web app in the background
nohup streamlit run frontend/app.py --server.port 8501 --server.address 0.0.0.0 > streamlit.log 2>&1 &
tail -f streamlit.log

# check that both are running
ps aux | grep uvicorn
ps aux | grep streamlit
```

Open port **8501** in the EC2 security group to use the web app. Port 8000 only needs to be open if you want the API itself to be public, because the web app can reach it at `127.0.0.1`. Folder names are case-sensitive on Linux, so use `frontend/app.py` and `Dataset`, exactly as written.

## Known limitations

- **The director matters most.** In this data, movies almost always have a director and TV shows almost never do. On the test set the model is right on 99.8% of titles that follow this pattern (1,251 titles) but only 55.2% of titles that break it (67 titles: a movie with no director or a TV show with one). Those titles are about 5% of the data but cause 30 of the 32 mistakes.
- **Time range.** The model was trained on titles added in 2008 to 2021 and released in 1925 to 2021. Predictions for dates outside this range are guesses. The app's Date Added box starts at today's date, so pick a date from 2008 to 2021.
- **Genre list in the app.** The model knows 27 genre columns, but the app offers 24 options. *Action & Adventure*, *Anime* and *British* cannot be selected, and the app's *Kids* option is sent as *Children & Family*, while the model also has a separate `Kids'` genre.
- Probabilities are model estimates, not guarantees. A result near 50% should be treated as a guess.

## Author

Built by Bharath Hottin as a machine learning internship project.
