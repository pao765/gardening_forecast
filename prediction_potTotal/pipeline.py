"""
UBICACIÓN: prediction_potTotal/pipeline.py

Clase Pipeline: corre los 8 experimentos que ya definiste en el
notebook (2 algoritmos x 2 configuraciones de variables x 2
representaciones del target), más el baseline de persistencia,
y arma la tabla comparativa final.
"""

import numpy as np
import pandas as pd

from prediction_potTotal.config import config
from prediction_potTotal.dataset import DataLoader
from prediction_potTotal.features import Preprocesador
from prediction_potTotal.metrics import Evaluador
from prediction_potTotal.modeling.models import (
    ModeloBaselinePersistencia,
    ModeloRandomForest,
    ModeloXGBoost,
)

COLUMNAS_CLIMA = [
    "mean_temp", "mean_humid", "mean_prec_height_mm",
    "total_prec_height_mm", "mean_sun_dur_min", "total_sun_dur_h",
]


class Pipeline:
    """Orquesta: carga -> preparación -> 8 experimentos -> comparación."""

    def __init__(self):
        self.cargador = DataLoader()
        self.evaluador = Evaluador()
        self.config = config
        self.horizonte = self.config.obtener("horizonte_dias")
        self.target_col = f"target_PotTotal_{self.horizonte}d"

    def preparar_datos(self):
        df = self.cargador.cargar_datos()
        df = self.cargador.crear_variable_objetivo_base(df)

        preprocesador = Preprocesador()
        dataset = preprocesador.preparar_dataset_modelado(df, COLUMNAS_CLIMA, self.horizonte)

        train, test = preprocesador.dividir_train_test_cronologico(dataset)
        return train, test

    def _construir_features(self, train, test, columnas_numericas):
        """Ajusta scaler/encoder SOLO con train, transforma ambos splits."""
        prep = Preprocesador()
        prep.ajustar_transformadores(train, columnas_numericas)
        X_train = prep.transformar(train)
        X_test = prep.transformar(test)
        return X_train, X_test

    def ejecutar_experimentos(self) -> pd.DataFrame:
        train, test = self.preparar_datos()

        y_train_raw = train[self.target_col].values
        y_test_raw = test[self.target_col].values
        y_train_log = np.log1p(y_train_raw)

        cols_lags = ["lag_7", "lag_14", "rolling_mean_7", "rolling_mean_14"]
        configuraciones = {
            "A (sin clima)": cols_lags,
            "B (con clima)": COLUMNAS_CLIMA + cols_lags,
        }
        algoritmos = {
            "Random Forest": lambda: ModeloRandomForest(
                n_estimadores=self.config.obtener("rf_n_estimadores"),
                profundidad_maxima=self.config.obtener("rf_profundidad_maxima"),
            ),
            "XGBoost": lambda: ModeloXGBoost(
                n_estimadores=self.config.obtener("xgb_n_estimadores"),
                profundidad_maxima=self.config.obtener("xgb_profundidad_maxima"),
                tasa_aprendizaje=self.config.obtener("xgb_tasa_aprendizaje"),
            ),
        }
        targets = {"original": (y_train_raw, None), "log": (y_train_log, "log")}

        resultados = []
        for nombre_cfg, cols_numericas in configuraciones.items():
            X_train, X_test = self._construir_features(train, test, cols_numericas)

            for nombre_target, (y_tr, modo) in targets.items():
                for nombre_algo, constructor in algoritmos.items():
                    modelo = constructor()
                    modelo.entrenar(X_train, y_tr)
                    pred = modelo.predecir(X_test)
                    if modo == "log":
                        pred = np.expm1(pred)

                    resultado = self.evaluador.evaluar(
                        y_test_raw, pred,
                        nombre=f"{nombre_algo} | {nombre_cfg} | {nombre_target}",
                    )
                    resultado["Algoritmo"] = nombre_algo
                    resultado["Configuración"] = nombre_cfg
                    resultado["Target"] = nombre_target
                    resultados.append(resultado)

        return self.evaluador.generar_reporte(resultados)

    def calcular_baseline(self) -> dict:
        _, test = self.preparar_datos()
        baseline = ModeloBaselinePersistencia()
        baseline.entrenar(None, None)
        pred_baseline = baseline.predecir(test)
        return self.evaluador.evaluar(test[self.target_col].values, pred_baseline,
                                       nombre="Baseline (persistencia)")

    def comparar_todo(self) -> pd.DataFrame:
        resultados = self.ejecutar_experimentos()
        return self.evaluador.comparar_experimentos(resultados)


if __name__ == "__main__":
    # Ejecuta con: python -m prediction_potTotal.pipeline
    pipeline = Pipeline()

    print("Calculando baseline de persistencia...")
    baseline = pipeline.calcular_baseline()
    print(baseline)

    print("\nCorriendo los 8 experimentos...")
    tabla = pipeline.comparar_todo()
    print(tabla[["Modelo", "MAE", "RMSE", "R2", "sMAPE (%)"]])

    mejor = tabla.iloc[0]
    mejora = pipeline.evaluador.mejora_vs_baseline(mejor.to_dict(), baseline)
    print(f"\nEl mejor experimento reduce el MAE del baseline en {mejora:.1f}%.")
