"""
UBICACIÓN: prediction_potTotal/modeling/predict.py

Carga el modelo, el preprocesador y la metadata ya guardados por
train.py, y genera predicciones reales de PotTotal a 'horizonte'
días vista, a partir del historial de ventas más reciente.

IMPORTANTE: para predecir de verdad hacia el futuro, este script
necesita el HISTORIAL de datos (no una sola fila) porque las
variables lag_7, lag_14 y las medias móviles dependen de días
anteriores. La predicción se genera para el día de origen t0 más
reciente que tenga historial suficiente (últimos `horizonte` días
del dataset, que no tienen target conocido todavía).

Se ejecuta con: python -m prediction_potTotal.modeling.predict
"""

import json

import joblib
import numpy as np
import pandas as pd

from prediction_potTotal.config import MODELS_DIR, RUTA_CASHIER_DATA
from prediction_potTotal.dataset import DataLoader
from prediction_potTotal.pipeline import COLUMNAS_CLIMA


class Predictor:
    """Envuelve la carga del modelo + preprocesador y la generación de predicciones."""

    def __init__(self, models_dir=MODELS_DIR):
        self.models_dir = models_dir
        self.modelo = None
        self.preprocesador = None
        self.metadata = None

    def cargar_artefactos(self):
        with open(self.models_dir / "metadata_modelo.json", encoding="utf-8") as f:
            self.metadata = json.load(f)

        self.preprocesador = joblib.load(self.models_dir / "preprocesador_pot_total_7d.pkl")

        # El modelo se guardó con joblib.dump(self.modelo, ...) dentro de
        # ModeloBase, así que aquí se carga directo el objeto sklearn/xgboost.
        self.modelo = joblib.load(self.models_dir / "modelo_pot_total_7d.pkl")
        return self

    def predecir_desde_historial(self, df_historico: pd.DataFrame, n_predicciones: int = 1) -> pd.DataFrame:
        """
        df_historico: DataFrame crudo con las mismas columnas de CashierData
        (Date, PotOwn, PotPurchased, variables de clima, feriados, etc.),
        incluyendo los días más recientes disponibles.

        n_predicciones: cuántos días de origen distintos predecir (por
        defecto, solo el más reciente).
        """
        if self.modelo is None or self.preprocesador is None:
            raise RuntimeError("Llama a cargar_artefactos() antes de predecir.")

        cargador = DataLoader()
        df = cargador.crear_variable_objetivo_base(df_historico)

        # dataset "de predicción": no exige target conocido, solo historial (lag_14)
        prep_temporal = self.preprocesador.__class__()
        dataset_prediccion = prep_temporal.preparar_dataset_prediccion(
            df, COLUMNAS_CLIMA, horizonte=self.metadata["horizonte_dias"]
        )

        origenes = dataset_prediccion.tail(n_predicciones).copy()
        X_nuevo = self.preprocesador.transformar(origenes)

        predicciones = self.modelo.predict(X_nuevo)
        if self.metadata["usa_log"]:
            predicciones = np.expm1(predicciones)

        resultado = pd.DataFrame({
            "fecha_origen": origenes["Date"].values,
            "fecha_objetivo": origenes["Date"].values + pd.Timedelta(days=self.metadata["horizonte_dias"]),
            "PotTotal_predicho": np.round(predicciones, 1),
        })
        return resultado


if __name__ == "__main__":
    cargador = DataLoader()
    df_historico = cargador.cargar_datos()

    predictor = Predictor().cargar_artefactos()
    resultado = predictor.predecir_desde_historial(df_historico, n_predicciones=5)

    print(f"Modelo usado: {predictor.metadata['algoritmo']} "
          f"(clima={predictor.metadata['usa_clima']}, log={predictor.metadata['usa_log']})")
    print(f"MAE en validación: {predictor.metadata['mae_validacion']}")
    print()
    print(resultado.to_string(index=False))
