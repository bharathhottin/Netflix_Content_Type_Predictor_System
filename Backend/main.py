from fastapi import FastAPI
from pydantic import BaseModel

from Backend.predictor import predict

app = FastAPI(
    title="Netflix Content Type Prection API"
)


# input schema
# input schema
class NetFlixContent(BaseModel):

    director    :   str
    date_added  :   str
    release_year:   int
    rating      :   str
    listed_in   :   str


@app.get("/type")
def type_check():
    return {"status": "ok"}    


# type prediction endpoint
@app.post("/netflix-type-prediction-system")
def predict_content_type(input_data: NetFlixContent):
    input_data = input_data.model_dump()
    result = predict(input_data=input_data)
    return{
        "prediction": result["prediction"],
        "probability": result["probability"],
        "content": (
            "TV Show 📺"
            if result["prediction"] == 1
            else "Movie 🎥"
        )
    }