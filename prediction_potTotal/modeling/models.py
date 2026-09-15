"""
UBICACIÓN: prediction_potTotal/modeling/models.py

ModeloBase abstracta + 3 implementaciones: Random Forest, XGBoost, y
un modelo de persistencia (baseline) que replica exactamente el
criterio de referencia que ya usas en el notebook: predecir con
lag_7, sin entrenamiento real.
"""

from abc import ABC, abstractmethod

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor


class ModeloBase(ABC):
    """Contrato común: entrenar, predecir, guardar y cargar."""

    def __init__(self, nombre_modelo: str):
        self.nombre_modelo = nombre_modelo
        self.esta_entrenado = False
        self.modelo = None

    @abstractmethod
    def entrenar(self, X, y):
        ...

    @abstractmethod
    def predecir(self, X):
        ...

    def guardar_modelo(self, ruta):
        joblib.dump(self.modelo, ruta)

    def cargar_modelo(self, ruta):
        self.modelo = joblib.load(ruta)
        self.esta_entrenado = True


class ModeloRandomForest(ModeloBase):
    def __init__(self, n_estimadores=200, profundidad_maxima=12, semilla_aleatoria=42):
        super().__init__(nombre_modelo="RandomForest")
        self.modelo = RandomForestRegressor(
            n_estimators=n_estimadores,
            max_depth=profundidad_maxima,
            random_state=semilla_aleatoria,
            n_jobs=-1,
        )

    def entrenar(self, X, y):
        self.modelo.fit(X, y)
        self.esta_entrenado = True

    def predecir(self, X):
        return self.modelo.predict(X)


class ModeloXGBoost(ModeloBase):
    def __init__(self, n_estimadores=150, profundidad_maxima=6,
                 tasa_aprendizaje=0.05, semilla_aleatoria=42):
        super().__init__(nombre_modelo="XGBoost")
        self.modelo = XGBRegressor(
            n_estimators=n_estimadores,
            max_depth=profundidad_maxima,
            learning_rate=tasa_aprendizaje,
            random_state=semilla_aleatoria,
            n_jobs=-1,
        )

    def entrenar(self, X, y):
        self.modelo.fit(X, y)
        self.esta_entrenado = True

    def predecir(self, X):
        return self.modelo.predict(X)


class ModeloBaselinePersistencia(ModeloBase):
    """
    Modelo de referencia (no requiere entrenamiento): predice el
    valor de PotTotal 7 días después como si fuera igual al valor
    registrado en el día de origen t0 (columna lag_7).
    """

    def __init__(self, columna_lag: str = "lag_7"):
        super().__init__(nombre_modelo="Baseline (persistencia)")
        self.columna_lag = columna_lag

    def entrenar(self, X, y):
        # No hay nada que entrenar: el "modelo" es la regla lag_7.
        self.esta_entrenado = True

    def predecir(self, X):
        return X[self.columna_lag].values
