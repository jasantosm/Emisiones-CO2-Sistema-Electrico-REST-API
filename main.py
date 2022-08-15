from fastapi import FastAPI
from typing import List
from pydantic import BaseModel
from fastapi import status

app = FastAPI()

class Data(BaseModel):

    Date: str 
    daily_generacion: float 
    daily_consumo_combustible: float
    daily_demanda: float
    daily_aportes_energia: float
    daily_volumen_util_energia: float
    daily_disponibilidad_real: float
    daily_precio_bolsa: float
    daily_emision_CO2_eq: float



@app.get("/model/k", response_model=List[Data], status_code=status.HTTP_200_OK)
def get_prediction(k: int, test_data: List[Data]):



    return test_data
