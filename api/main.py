import json
import pickle
import os
from smart_open import open
import shap



import pandas as pd
from fastapi import FastAPI, Body, HTTPException

from sample_call import test_dict

th_proba = 0.54
model_path = os.getenv('MODEL_PATH', 'tests/model.pkl')
with open(model_path, 'rb') as file:
    model = pickle.load(file)

explainer = shap.TreeExplainer(model)

app = FastAPI()


@app.post("/predict", status_code=200)
def predict_credit(request: dict = Body(examples=[test_dict])):
    df = pd.DataFrame([request])

    try:
        output_proba = float(model.predict_proba(df)[:, 1][0])
        output = int(model.predict(df)[0])
    except Exception as e:
        print(e)
        raise HTTPException(status_code=400, detail=str(e))

    if output_proba >= th_proba:
        prediction = "crédit refusé"
    else:
        prediction = "crédit accordé"

    results = {'credit_score_risk':
                   {'predict_proba': output_proba,
                    'predict_business_risk': output,
                    'predict_th_proba': th_proba},
               'prediction': prediction
               }

    json_results = json.dumps(results)
    return json_results


@app.post("/contrib", status_code=200)
def predict_contrib(request: dict = Body(examples=[test_dict])):
    df = pd.DataFrame([request])

    try:
        shap_values = explainer.shap_values(df)
    except Exception as e:
        print(e)
        raise HTTPException(status_code=400, detail=str(e))

    results = {'shap_values': list(shap_values[1][0])}

    json_results = json.dumps(results)
    return json_results


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)
