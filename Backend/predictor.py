import os
import logging
import re
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from joblib import load

# load .env content to env var
load_dotenv()

PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT")).resolve()
MODEL_PATH = PROJECT_ROOT / os.getenv("MODEL_DIR") / os.getenv("MODEL_NAME")
LOG_PATH = PROJECT_ROOT / os.getenv("LOG_DIR") / os.getenv("LOG_NAME")

LOG_PATH.parent.mkdir(parents=True, exist_ok=True)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_PATH)
    ]
)

# load trained model only once
model = load(MODEL_PATH)
logging.info("Model loaded sucessfully...")

LISTIN_TYPE = {
    "Docuseries": "Documentaries",
    "Kids": "Children & Family",
    "Stand-Up Comedy & Talk": "Stand-Up Comedy",
    "Classic": "Classic & Cult",
    "Cult": "Classic & Cult",
}

def cleaned_listin(text):
    names = []
    for g in text.split(","):
        g = re.sub(r"\b(TV|Movies|Shows|Series|Features)\b", "", g)
        g = " ".join(g.split())
        g = LISTIN_TYPE.get(g, g)
        if g:
            names.append(g)
    return sorted(set(names))


def predict(input_data:dict):

    features = {
        "release_year": input_data["release_year"],
        "year_added": pd.to_datetime(input_data["date_added"].strip(), format="%m/%d/%Y").year,
        "has_director": int(input_data["director"].strip() != "Not Given"),
        f"rating_{input_data['rating'].strip()}": 1,
    }
    for g in cleaned_listin(input_data["listed_in"]):
        features["listed_" + g] = 1

    # same columns, same order as training; anything missing becomes 0
    df = pd.DataFrame([features]).reindex(columns=model.feature_names_in_, fill_value=0)


    # get the prediction
    prediction = int(model.predict(df)[0])

    # get the probability
    probability = float(model.predict_proba(df)[0][1])

    logging.info(f"Model provided prediction: {prediction}, Probability: {probability}")

    return{
        "prediction": prediction,
        "probability": probability
    }


# # example usage
# sample_input = {
#     "director": "Not Given",
#     "release_year": 2020,
#     "date_added": "7/31/2020",
#     "rating": "TV-PG",
#     "listed_in": "British TV Shows, Crime TV Shows, International TV Shows",
# }
# result = predict(input_data=sample_input)
# print(result)
