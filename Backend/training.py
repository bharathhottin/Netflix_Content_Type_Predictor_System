import os
import logging
import re
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from joblib import dump

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    recall_score,
)


def train_model():
    try:

        # load the .env file content
        load_dotenv()

        PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT")).resolve()

        DATASET_PATH = PROJECT_ROOT / os.getenv("DATASET_DIR") / os.getenv("DATASET_NAME")
        MODEL_PATH = PROJECT_ROOT / os.getenv("MODEL_DIR") / os.getenv("MODEL_NAME")
        LOG_PATH = PROJECT_ROOT / os.getenv("LOG_DIR") / os.getenv("LOG_NAME")

        TARGET_COL = os.getenv("TARGET_COL")
        TEST_SIZE = float(os.getenv("TEST_SIZE"))
        RANDOM_STATE = int(os.getenv("RANDOM_STATE"))

        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)


        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(message)s",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(LOG_PATH)
            ]
        )

        # loading the dataset
        df = pd.read_csv(DATASET_PATH)
        logging.info(f"Dataset read Sucessfully with shape of: {df.shape}")

        # check for duplicates
        for col in df.select_dtypes(exclude="number").columns:
            df[col] = df[col].str.strip()                                   # remove stray spaces
        before = len(df)
        df = df.drop_duplicates(subset=df.columns.drop("show_id"))          # same title listed twice
        df = df.reset_index(drop=True)
        print(f"Rows: {before:,} -> {len(df):,}  ({before - len(df)} duplicates removed)")
        df["type"].value_counts()

        # droping id column
        df=df.drop(columns=["show_id"])

        # encode target: Movie -> 0, TV Show -> 1 (same as the notebook)
        y = df[TARGET_COL].map({"Movie": 0, "TV Show": 1})
        y = y.astype(int)

        LISTIN_TYPE = {
            "Docuseries": "Documentaries",
            "Kids": "Children & Family",
            "Stand-Up Comedy & Talk": "Stand-Up Comedy",
            "Classic": "Classic & Cult",
            "Cult" : "Classic & Cult"
        }

        def cleaned_listin(text):
            names = []
            for g in text.split(","):
                g = re.sub(r"\b(TV|Movies|Shows|Series|Features)\b", "", g)
                g = " ".join(g.split())
                g = LISTIN_TYPE.get(g, g)
                if g:
                    names.append(g)
            return ",".join(sorted(set(names))) 

        New_Listed_in = df["listed_in"].apply(cleaned_listin).str.get_dummies(sep=",").drop(columns="", errors="ignore") 
        New_Listed_in.columns = ["listed_" + c for c in New_Listed_in.columns]          


        X = pd.DataFrame(index=df.index)
        X["release_year"] = df["release_year"]
        X["year_added"] = pd.to_datetime(df["date_added"], format="%m/%d/%Y").dt.year
        X["has_director"] = (df["director"] != "Not Given").astype(int)

        rating = pd.get_dummies(df["rating"], prefix="rating", dtype=int)  # one-hot encoding 

        X = pd.concat([X, rating, New_Listed_in], axis=1)
        print("Feature table:", X.shape[0], "rows x", X.shape[1], "columns")
        print("Missing values:", int(X.isna().sum().sum()))
        print("Columns: 3 numbers +", rating.shape[1], "ratings +", New_Listed_in.shape[1], "Lists")


        # train test split
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            random_state=RANDOM_STATE,
            stratify=y,
            test_size=TEST_SIZE
        )

        logging.info(f"Training Shape: {X_train.shape},  Test Shape: {X_test.shape}")

        # best parameters for the trained moel
        best_model = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("model", SVC(
                    random_state=RANDOM_STATE,
                    C= 10,
                    gamma= 0.01,
                    probability=True
                ))
            ]
        )

        best_model.fit(X_train, y_train)
        logging.info(f"Model training Completed")

        y_train_pred = best_model.predict(X_train)
        y_test_pred = best_model.predict(X_test)

        # accuracy
        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)

        # recall score
        train_recall = recall_score(y_train, y_train_pred)
        test_recall = recall_score(y_test, y_test_pred)
        
        # f1 score
        train_f1 = f1_score(y_train, y_train_pred)
        test_f1 = f1_score(y_test, y_test_pred)

        logging.info(f"Training Accuracy: {train_acc:.4f}  | Recall: {train_recall:.4f}   | F1: {train_f1:.4f}")
        logging.info(f"Test Accuracy: {test_acc:.4f}  | Recall: {test_recall:.4f}   | F1: {test_f1:.4f}")

        target_names = ["Movie", "TV Show"]
        logging.info(f"Training Classification Report:\n" + classification_report(y_train, y_train_pred, target_names=target_names))
        logging.info(f"Test Classification Report:\n" + classification_report(y_test, y_test_pred, target_names=target_names))


        # save the trained model
        dump(best_model, MODEL_PATH)
        logging.info(f"Model saved to: {MODEL_PATH}")

        logging.info("Training Script Completed")

    except Exception as e:
        print(f"Training Failed: {e}") 
        logging.exception(f"Training Script Failed: {e}")
        raise


if __name__ == "__main__"  :
    train_model()     