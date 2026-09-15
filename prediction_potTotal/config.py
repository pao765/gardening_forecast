"""
UBICACIÓN: prediction_potTotal/config.py

Guarda en un solo lugar las rutas y parámetros del proyecto, igual
que en el proyecto Walmart, para no repetir valores "quemados" por
todo el código.
"""

from pathlib import Path

PROJ_ROOT = Path(__file__).resolve().parents[1]

# --- Rutas de datos ---
DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

RUTA_CASHIER_DATA = RAW_DATA_DIR / "CashierData.csv"

# --- Rutas de modelos entrenados ---
MODELS_DIR = PROJ_ROOT / "models"

# --- Parámetros del proyecto ---
class ConfiguracionProyecto:
    """
    Centraliza los parámetros que se repiten en varias partes del
    pipeline: horizonte de pronóstico, hiperparámetros de los modelos,
    y configuración de la división train/test.
    """

    def __init__(self):
        self.parametros = {
            "horizonte_dias": 7,
            "pct_test": 0.2,
            "random_state": 42,
            # Random Forest
            "rf_n_estimadores": 200,
            "rf_profundidad_maxima": 12,
            # XGBoost
            "xgb_n_estimadores": 150,
            "xgb_profundidad_maxima": 6,
            "xgb_tasa_aprendizaje": 0.05,
        }

    def obtener(self, clave):
        return self.parametros.get(clave)


config = ConfiguracionProyecto()
