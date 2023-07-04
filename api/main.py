import json
import pickle
from smart_open import open

import pandas as pd
from fastapi import FastAPI, Body, HTTPException

from sample_call import test_dict

th_proba = 0.54
model_path = '/Users/audreyhohmann/Documents/Formation/OCR/P7/mlruns/274439274391636157/99fce0a77f4045eda1e8cb7e14c57ab7/artifacts/model/model.pkl'
with open(model_path, 'rb') as file:
    model = pickle.load(file)

app = FastAPI()


@app.post("/", status_code=200)
def predict_credit(request: dict = Body(examples=[test_dict])):
    df = pd.DataFrame([request])

    try:
        output_proba = float(model.predict_proba(df)[:, 1][0])
        output = int(model.predict(df)[0])
    except Exception as e:
        print(e)
        raise HTTPException(status_code=400, detail=str(e))

    if output_proba >= th_proba:
        prediction = "crédit non accordé"
    else:
        prediction = "crédit accordé"

    results = {'credit_score_risk':
                   {'predict_proba': output_proba,
                    'predict_business_risk': output},
               'prediction': prediction
               }

    json_results = json.dumps(results)
    return json_results


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)
