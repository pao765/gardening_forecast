"""
UBICACIÓN: prediction_potTotal/dataset.py

Clase DataLoader: solo lee el CSV crudo desde disco. No transforma
nada — eso es responsabilidad de Preprocesador (features.py).
"""

import pandas as pd

from prediction_potTotal.config import RUTA_CASHIER_DATA


class DataLoader:
    """Carga CashierData.csv respetando su formato europeo (';' y ',')."""

    def __init__(self, ruta_datos=RUTA_CASHIER_DATA):
        self.ruta_datos = ruta_datos

    def cargar_datos(self) -> pd.DataFrame:
        df = pd.read_csv(self.ruta_datos, sep=";", decimal=",")
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").reset_index(drop=True)
        return df

    def crear_variable_objetivo_base(self, df: pd.DataFrame) -> pd.DataFrame:
        """PotTotal = PotOwn + PotPurchased (variable objetivo del proyecto)."""
        df = df.copy()
        df["PotTotal"] = df["PotOwn"] + df["PotPurchased"]
        return df


if __name__ == "__main__":
    # Prueba rápida: python -m prediction_potTotal.dataset
    cargador = DataLoader()
    df = cargador.cargar_datos()
    df = cargador.crear_variable_objetivo_base(df)
    print(f"Dataset cargado: {df.shape[0]} filas x {df.shape[1]} columnas")
    print(f"Rango de fechas: {df['Date'].min().date()} a {df['Date'].max().date()}")
