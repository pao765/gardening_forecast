"""
UBICACIÓN: prediction_potTotal/modeling/train.py

Corre el pipeline completo, identifica el experimento ganador por
MAE, lo re-entrena y guarda TANTO el modelo final COMO el
Preprocesador ya ajustado (scaler + encoder) — sin este último,
predict.py no podría transformar datos nuevos de la misma forma que
se transformó el conjunto de entrenamiento.

Se ejecuta con: python -m prediction_potTotal.modeling.train
"""

import json

import joblib
import numpy as np

from prediction_potTotal.config import MODELS_DIR
from prediction_potTotal.features import Preprocesador
from prediction_potTotal.modeling.models import ModeloRandomForest, ModeloXGBoost
from prediction_potTotal.pipeline import COLUMNAS_CLIMA, Pipeline


def main():
    pipeline = Pipeline()
    tabla = pipeline.comparar_todo()
    mejor = tabla.iloc[0]
    print(f"Mejor experimento: {mejor['Modelo']} (MAE={mejor['MAE']})")

    train, test = pipeline.preparar_datos()
    y_train_raw = train[pipeline.target_col].values

    cols_lags = ["lag_7", "lag_14", "rolling_mean_7", "rolling_mean_14"]
    usa_clima = "con clima" in mejor["Configuración"]
    cols_numericas = COLUMNAS_CLIMA + cols_lags if usa_clima else cols_lags

    # Se ajusta un Preprocesador NUEVO (no el interno de pipeline, que se
    # descarta) para poder guardarlo y reutilizarlo después en predict.py
    preprocesador_final = Preprocesador()
    preprocesador_final.ajustar_transformadores(train, cols_numericas)
    X_train = preprocesador_final.transformar(train)

    usa_log = mejor["Target"] == "log"
    y_final = np.log1p(y_train_raw) if usa_log else y_train_raw

    modelo_final = (
        ModeloRandomForest() if mejor["Algoritmo"] == "Random Forest" else ModeloXGBoost()
    )
    modelo_final.entrenar(X_train, y_final)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    modelo_final.guardar_modelo(MODELS_DIR / "modelo_pot_total_7d.pkl")
    joblib.dump(preprocesador_final, MODELS_DIR / "preprocesador_pot_total_7d.pkl")

    metadata = {
        "algoritmo": mejor["Algoritmo"],
        "usa_clima": usa_clima,
        "usa_log": usa_log,
        "columnas_numericas": cols_numericas,
        "horizonte_dias": pipeline.horizonte,
        "mae_validacion": float(mejor["MAE"]),
    }
    with open(MODELS_DIR / "metadata_modelo.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"Modelo guardado en {MODELS_DIR / 'modelo_pot_total_7d.pkl'}")
    print(f"Preprocesador guardado en {MODELS_DIR / 'preprocesador_pot_total_7d.pkl'}")
    print(f"Metadata guardada en {MODELS_DIR / 'metadata_modelo.json'}")


if __name__ == "__main__":
    main()
