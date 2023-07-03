import json
import pickle
from smart_open import open

import pandas as pd
from fastapi import FastAPI

model_path = '/Users/audreyhohmann/Documents/Formation/OCR/P7/mlruns/274439274391636157/99fce0a77f4045eda1e8cb7e14c57ab7/artifacts/model/model.pkl'
with open(model_path, 'rb') as file:
    model = pickle.load(file)

app = FastAPI()


@app.post("/")
def predict_credit(request: dict):
    df = pd.DataFrame([request])
    output_proba = model.predict_proba(df)
    output = model.predict(df)
    results = {'credit_score_risk':
                   {'predict_proba': float(output_proba[:, 1][0]),
                    'predict_business_risk': int(output[0])}
               }

    json_results = json.dumps(results)
    return json_results


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)
