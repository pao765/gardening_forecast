"""
UBICACIÓN: src/api/model_loader.py

Carga el modelo de predicción de PotTotal:
- En local: desde MLflow Model Registry (alias 'produccion')
- En Docker: desde el archivo .pkl local

También carga el preprocesador entrenado y la metadata.
"""

import os
import sys
import json
from pathlib import Path

# Permite importar prediction_potTotal estando en src/api/
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import joblib
import mlflow
import mlflow.sklearn

from prediction_potTotal.config import MODELS_DIR
from prediction_potTotal.features import Preprocesador

MODEL_NAME = "PotTotal_Horticola_Model"
ALIAS_PRODUCCION = "produccion"
TRACKING_URI = "sqlite:///mlflow.db"

# Detectar si estamos en Docker
IS_DOCKER = os.environ.get("ENVIRONMENT") == "docker"

if not IS_DOCKER:
    mlflow.set_tracking_uri(TRACKING_URI)


def cargarModelo():
    """
    Carga el modelo:
    - Local: desde MLflow Registry usando alias 'produccion'
    - Docker: desde el .pkl local
    """
    if IS_DOCKER:
        try:
            return joblib.load(MODELS_DIR / "modelo_pot_total_7d.pkl")
        except Exception:
            return None
    try:
        model_uri = f"models:/{MODEL_NAME}@{ALIAS_PRODUCCION}"
        return mlflow.sklearn.load_model(model_uri)
    except Exception:
        # Fallback: cargar desde .pkl si no está en Registry
        try:
            return joblib.load(MODELS_DIR / "modelo_pot_total_7d.pkl")
        except Exception:
            return None


def obtenerMetadataModelo() -> dict:
    """Version y run_id del modelo actual, para trazabilidad."""
    if IS_DOCKER:
        return {"version": "docker-pkl", "run_id": "n/a"}
    try:
        client = mlflow.tracking.MlflowClient()
        mv = client.get_model_version_by_alias(MODEL_NAME, ALIAS_PRODUCCION)
        return {"version": str(mv.version), "run_id": mv.run_id}
    except Exception:
        # Fallback: leer metadata del JSON local
        try:
            with open(MODELS_DIR / "metadata_modelo.json", "r", encoding="utf-8") as f:
                meta = json.load(f)
            return {
                "version": "local",
                "run_id": meta.get("algoritmo", "desconocido"),
            }
        except Exception:
            return {"version": "Desconocida", "run_id": "Desconocido"}


def obtenerPreprocesadorEntrenado() -> Preprocesador:
    """Carga el preprocesador ya ajustado en entrenamiento."""
    try:
        preprocesador = joblib.load(MODELS_DIR / "preprocesador_pot_total_7d.pkl")
        return preprocesador
    except Exception:
        return None