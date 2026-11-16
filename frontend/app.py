import os
import requests
import streamlit as st
from dotenv import load_dotenv


# loading .env to env vars
load_dotenv()

API_URL = os.getenv("API_URL")

st.set_page_config(
    page_title="Netflix Content Predictor",
    page_icon="🍿",
    layout="centered"
)

st.title("🍿 Netflix Content Predictor")
st.write("Netflix Content Type Predictor")

st.subheader("Enter the content details and click **Predict**")

ratings_list = [ "rating_G" , "rating_NC-17" , "rating_NR" ,  "rating_PG" , "rating_PG-13", "rating_R",  "rating_TV-14", "rating_TV-G", "rating_TV-MA", "rating_TV-PG", "rating_TV-Y", "rating_TV-Y7", "rating_TV-Y7-FV", "rating_UR"]
ratings = [c.removeprefix("rating_") for c in ratings_list if c.startswith("rating_")]

listed_list = ['listed_Children & Family', 'listed_Classic & Cult', 'listed_Comedies','listed_Crime', 'listed_Documentaries', 'listed_Dramas', 'listed_Faith & Spirituality', 'listed_Horror', 'listed_Independent', 'listed_International', 'listed_Kids', 'listed_Korean', 'listed_LGBTQ', 'listed_Music & Musicals', 'listed_Mysteries', 'listed_Reality', 'listed_Romantic', 'listed_Sci-Fi & Fantasy', 'listed_Science & Nature', 'listed_Spanish-Language', 'listed_Sports', 'listed_Stand-Up Comedy', 'listed_Teen', 'listed_Thrillers']

list_in = [c.removeprefix("listed_")for c in listed_list if c.startswith("listed_")]

col1, col2 = st.columns(2)

with col1:
    director = st.text_input("Director",  help="Leave empty if the title has no director")
    date_added = st.date_input("Date Added")
    release_year = st.number_input("Release Year", min_value=1900, max_value=2030, value=2020, step=1)


with col2:
    rating = st.selectbox("Rating", ratings)
    listed_in = st.multiselect("Listed in", list_in)


if st.button("🔍Predict"):
    if not listed_in:
        st.warning("Please select at least one 'Listed in' category.")
        st.stop()

    input_data = {
        "director": director.strip() or "Not Given",
        "date_added": date_added.strftime("%m/%d/%Y"),
        "release_year": int(release_year),
        "rating": rating,
        "listed_in": ", ".join(listed_in),
    }

    response = requests.post(API_URL, json=input_data)

    if response.status_code != 200:
        st.error("❌Something went wrong. Try again later...")

    else:
        result = response.json()
        prediction = result["prediction"]
        probability = result["probability"]
        content = result["content"]

        st.divider()

        st.metric(
            label="Content Type probability",
            value=f"{probability:.2f}"
        )

        if prediction == 1:
            st.error(f"Model Predicted: {content} ")
        else:
            st.success(f"Model predicted: {content} ")    