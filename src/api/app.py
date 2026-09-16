"""
UBICACIÓN: src/api/app.py

API REST con FastAPI para servir el modelo de predicción
de PotTotal a 7 días.

Se ejecuta con: uvicorn src.api.app:app --reload
"""

import sys
from pathlib import Path

# Permite importar prediction_potTotal
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.api.model_loader import (
    MODEL_NAME,
    cargarModelo,
    obtenerMetadataModelo,
    obtenerPreprocesadorEntrenado,
)

app = FastAPI(
    title="API de Predicción PotTotal Hortícola",
    description="API para predecir PotTotal a 7 días",
    version="1.0.0",
)

# Variables globales
modelo = None
preprocesador = None
metadataModelo = {"version": "Desconocida", "run_id": "Desconocido"}


@app.on_event("startup")
def eventoInicio():
    """Carga el modelo y el preprocesador UNA sola vez al arrancar."""
    global modelo, preprocesador, metadataModelo
    modelo = cargarModelo()
    preprocesador = obtenerPreprocesadorEntrenado()
    metadataModelo = obtenerMetadataModelo()
    
    print("✅ API iniciada")
    print(f"📊 Modelo: {MODEL_NAME}")
    print(f"📌 Versión: {metadataModelo['version']}")
    print(f"🔗 Run ID: {metadataModelo['run_id']}")


# ===== MODELOS DE DATOS (PYDANTIC) =====

class PotTotalEntrada(BaseModel):
    """Datos de entrada para una predicción."""
    mean_temp: float = Field(..., examples=[25.5], description="Temperatura media (°C)")
    mean_humid: float = Field(..., examples=[65.0], description="Humedad media (%)")
    mean_prec_height_mm: float = Field(..., examples=[2.3], description="Precipitación media (mm)")
    total_prec_height_mm: float = Field(..., examples=[15.0], description="Precipitación total (mm)")
    mean_sun_dur_min: float = Field(..., examples=[300.0], description="Duración media del sol (min)")
    total_sun_dur_h: float = Field(..., examples=[5.0], description="Duración total del sol (h)")
    lag_7: float = Field(..., examples=[1500.0], description="PotTotal hace 7 días")
    lag_14: float = Field(..., examples=[1450.0], description="PotTotal hace 14 días")
    rolling_mean_7: float = Field(..., examples=[1480.0], description="Media móvil 7 días")
    rolling_mean_14: float = Field(..., examples=[1470.0], description="Media móvil 14 días")
    public_holiday: str = Field(..., examples=["no"], description="Feriado público")
    school_holiday: str = Field(..., examples=["no"], description="Vacaciones escolares")
    weekday: str = Field(..., examples=["Monday"], description="Día de la semana")


class PrediccionRequest(BaseModel):
    """Lista de entradas para predecir."""
    datos: list[PotTotalEntrada]


class PrediccionResultado(BaseModel):
    """Resultado de una predicción."""
    pot_total_predicho: float
    indice: int


class ModeloMetadata(BaseModel):
    """Metadata del modelo."""
    nombre: str
    version: str
    run_id: str


class PrediccionResponse(BaseModel):
    """Respuesta completa de la API."""
    modelo_metadata: ModeloMetadata
    total_predicciones: int
    resultados: list[PrediccionResultado]


class EstadoResponse(BaseModel):
    """Estado de la API."""
    estado: str
    modelo: str
    version_produccion: str
    run_id: str


# ===== ENDPOINTS =====

@app.get("/", response_model=EstadoResponse)
def raiz():
    """Endpoint raíz - estado de la API."""
    return {
        "estado": "En línea",
        "modelo": MODEL_NAME,
        "version_produccion": metadataModelo["version"],
        "run_id": metadataModelo["run_id"],
    }


@app.get("/health")
def health():
    """Verifica que la API y el modelo están funcionando."""
    return {
        "status": "ok",
        "modelo_cargado": modelo is not None,
        "preprocesador_cargado": preprocesador is not None,
    }


@app.get("/metadata")
def get_metadata():
    """Devuelve la metadata del modelo."""
    return metadataModelo


@app.post("/predecir", response_model=PrediccionResponse)
def predecir(payload: PrediccionRequest):
    """
    Hace predicciones de PotTotal a 7 días.
    
    Envía una lista de datos climáticos y lags, devuelve las predicciones.
    """
    if modelo is None:
        raise HTTPException(status_code=500, detail="El modelo no está cargado.")
    if preprocesador is None:
        raise HTTPException(status_code=500, detail="El preprocesador no está cargado.")

    try:
        # 1. Convertir input a DataFrame
        filas = [dato.model_dump() for dato in payload.datos]
        df = pd.DataFrame(filas)

         # 2. Calcular columnas que el preprocesador necesita
        #    is_weekend: 1 si es sábado o domingo
        df["is_weekend"] = df["weekday"].isin(["Saturday", "Sunday"]).astype(int)
        
        #    is_holiday: 1 si es feriado público
        df["is_holiday"] = (df["public_holiday"] != "no").astype(int)
        
        #    month: mes del año (usa la fecha actual o pídela en el input)
        from datetime import datetime
        df["month"] = datetime.now().month

        
        # 2. Transformar con el preprocesador
        #    (el preprocesador ya está ajustado, solo transforma)
        X = preprocesador.transformar(df)

        # 3. Predecir
        predicciones = modelo.predict(X)

        # 4. Si usa log, invertir
        #    (esto lo puedes detectar desde metadata si lo guardaste)
        #    Por ahora asumimos que no usa log

        # 5. Armar resultados
        resultados = [
            {
                "indice": i,
                "pot_total_predicho": round(float(pred), 2),
            }
            for i, pred in enumerate(predicciones)
        ]

        return {
            "modelo_metadata": {
                "nombre": MODEL_NAME,
                "version": metadataModelo["version"],
                "run_id": metadataModelo["run_id"],
            },
            "total_predicciones": len(resultados),
            "resultados": resultados,
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error durante la inferencia: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)