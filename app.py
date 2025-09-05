from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import uvicorn
from src.pipeline.predict_pipeline import PredictPipeline

app = FastAPI(title="Cyberbullying Detection App")

# Setup templates folder
templates = Jinja2Templates(directory="templates")

# Load prediction pipeline (you can change model_type to 'partial_ft' or 'lora')
pipeline = PredictPipeline(model_type="full_ft")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """
    Serve the homepage with input form
    """
    return templates.TemplateResponse("index.html", {"request": request, "prediction": None})


@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, comment: str = Form(...)):
    """
    Receive comment from form and return prediction
    """
    predictions = pipeline.predict([comment])
    result = predictions[0]  # single comment

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "prediction": result["predicted_label"],
            "probabilities": result["probabilities"],
            "comment": comment
        }
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
