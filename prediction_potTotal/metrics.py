"""
UBICACIÓN: prediction_potTotal/metrics.py

Clase Evaluador: las mismas 4 métricas que ya definiste en el
notebook (MAE, RMSE, R², sMAPE), más utilidades para comparar
los 8 experimentos entre sí.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class Evaluador:
    """Calcula métricas de evaluación y arma reportes comparativos."""

    def smape(self, y_true, y_pred) -> float:
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        denominador = (np.abs(y_true) + np.abs(y_pred)) / 2
        diferencia = np.abs(y_true - y_pred)
        resultado = np.divide(diferencia, denominador, out=np.zeros_like(diferencia),
                               where=denominador != 0)
        return 100 * np.mean(resultado)

    def evaluar(self, y_true, y_pred, nombre: str) -> dict:
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        s = self.smape(y_true, y_pred)
        return {
            "Modelo": nombre,
            "MAE": round(mae, 2),
            "RMSE": round(rmse, 2),
            "R2": round(r2, 3),
            "sMAPE (%)": round(s, 2),
        }

    def generar_reporte(self, lista_resultados: list) -> pd.DataFrame:
        return pd.DataFrame(lista_resultados)

    def comparar_experimentos(self, resultados_df: pd.DataFrame) -> pd.DataFrame:
        return resultados_df.sort_values("MAE").reset_index(drop=True)

    def mejora_vs_baseline(self, resultado_mejor: dict, resultado_baseline: dict) -> float:
        return (1 - resultado_mejor["MAE"] / resultado_baseline["MAE"]) * 100
