from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from cp import ChainedPredictor

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

predictor = ChainedPredictor() # type: ignore

@app.post("/predict")
async def predict(request: Request):
    body = await request.json()
    area = body.get("area")
    time = body.get("time")
    result = predictor.predict(area, time) # type: ignore
    return result
