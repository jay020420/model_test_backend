import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from .schemas import PredictRequest, PredictResponse, ParseRequest, ParseResponse
from .service import predict_batch, quickscore
from .nlp import parse_utterance

app = FastAPI(title="SME Early Warning API", version="1.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/nlp/parse", response_model=ParseResponse)
def nlp_parse(req: ParseRequest):
    return ParseResponse(**parse_utterance(req.utterance))


@app.post("/quickscore", response_model=PredictResponse)
def quick_score(req: PredictRequest):
    return PredictResponse(**quickscore(req.dict()))


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    res = predict_batch(req.store_id, req.target_month)
    if res is None:
        if any([req.sales_1m, req.sales_3m_avg, req.cust_1m, req.cust_3m_avg]):
            return PredictResponse(**quickscore(req.dict()))
        raise HTTPException(status_code=404, detail="No prediction found.")
    return PredictResponse(**res)


@app.get("/healthz")
def health():
    return {"status": "ok", "base_dir": os.environ.get("BASE_DIR")}