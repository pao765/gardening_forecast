"""
UBICACIÓN: prediction_potTotal/features.py

Clase Preprocesador: replica exactamente las decisiones tomadas en
el notebook de preparación de datos.

Punto crítico (igual que en el notebook): como se pronostica 7 días
hacia adelante, los lags NO pueden ser lag_1 (fuga de información).
Se usan lag_7 y lag_14, calculados ANTES de desplazar la variable
objetivo con .shift(-horizonte).
"""

import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from prediction_potTotal.config import config


class Preprocesador:
    """Limpieza, feature engineering y partición train/test para PotTotal."""

    def __init__(self):
        self.scaler = None
        self.encoder = None
        self.columnas_numericas = None
        self.columnas_categoricas = ["school_holiday", "weekday"]

    # --- Selección y verificación ---

    def seleccionar_variables(self, df: pd.DataFrame, columnas_clima: list) -> pd.DataFrame:
        columnas_calendario = ["public_holiday", "school_holiday"]
        prep = df[["Date", "PotTotal"] + columnas_clima + columnas_calendario].copy()
        return prep.sort_values("Date").reset_index(drop=True)

    def verificar_inconsistencias(self, df: pd.DataFrame) -> dict:
        inconsistencias = {
            "PotTotal_negativos": int((df["PotTotal"] < 0).sum()),
            "humedad_fuera_rango": int(((df["mean_humid"] < 0) | (df["mean_humid"] > 100)).sum()),
            "precipitacion_negativa": int((df["mean_prec_height_mm"] < 0).sum()),
        }
        return inconsistencias

    # --- Valores faltantes ---

    def tratar_valores_faltantes(self, df: pd.DataFrame, columnas_clima: list) -> pd.DataFrame:
        """Interpolación lineal temporal — apropiada para huecos cortos en variables climáticas."""
        df = df.copy()
        df[columnas_clima] = df[columnas_clima].interpolate(method="linear", limit_direction="both")
        return df

    # --- Feature engineering ---

    def crear_variables_calendario(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["weekday"] = df["Date"].dt.day_name()
        df["month"] = df["Date"].dt.month
        df["is_weekend"] = df["weekday"].isin(["Saturday", "Sunday"]).astype(int)
        df["is_holiday"] = (df["public_holiday"] != "no").astype(int)
        return df

    def crear_variables_lag_y_objetivo(self, df: pd.DataFrame, horizonte: int = None) -> pd.DataFrame:
        """
        Crea lag_7, lag_14, medias móviles, y la variable objetivo
        desplazada 'horizonte' días hacia adelante. El orden importa:
        los lags se calculan ANTES del shift del target.
        """
        horizonte = horizonte or config.obtener("horizonte_dias")
        df = df.copy()

        df["lag_7"] = df["PotTotal"]
        df["lag_14"] = df["PotTotal"].shift(7)
        df["rolling_mean_7"] = df["PotTotal"].rolling(window=7).mean()
        df["rolling_mean_14"] = df["PotTotal"].rolling(window=14).mean()

        df[f"target_PotTotal_{horizonte}d"] = df["PotTotal"].shift(-horizonte)
        return df

    # --- Partición y transformadores (fit SOLO en train) ---

    def dividir_train_test_cronologico(self, df: pd.DataFrame, pct_test: float = None):
        pct_test = pct_test if pct_test is not None else config.obtener("pct_test")
        n = len(df)
        corte = int(n * (1 - pct_test))
        return df.iloc[:corte].copy(), df.iloc[corte:].copy()

    def ajustar_transformadores(self, X_train: pd.DataFrame, columnas_numericas: list) -> None:
        """Ajusta scaler y encoder SOLO con el conjunto de entrenamiento."""
        self.columnas_numericas = columnas_numericas
        self.scaler = StandardScaler().fit(X_train[columnas_numericas])
        self.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        self.encoder.fit(X_train[self.columnas_categoricas])

    def transformar(self, X_split: pd.DataFrame) -> pd.DataFrame:
        """Aplica los transformadores YA ajustados (nunca reajusta)."""
        if self.scaler is None or self.encoder is None:
            raise RuntimeError("Llama a ajustar_transformadores() antes de transformar().")

        num = pd.DataFrame(
            self.scaler.transform(X_split[self.columnas_numericas]),
            columns=self.columnas_numericas, index=X_split.index,
        )
        cat = pd.DataFrame(
            self.encoder.transform(X_split[self.columnas_categoricas]),
            columns=self.encoder.get_feature_names_out(self.columnas_categoricas),
            index=X_split.index,
        )
        otras = X_split[["is_weekend", "is_holiday", "month"]]
        return pd.concat([num, cat, otras], axis=1)

    # --- Método orquestador ---

    def preparar_dataset_modelado(self, df: pd.DataFrame, columnas_clima: list,
                                   horizonte: int = None) -> pd.DataFrame:
        """Corre todo el pipeline de preparación y devuelve el dataset listo para X/y.
        Requiere que el target sea conocido (dropna también sobre el target) —
        úsalo para ENTRENAR y EVALUAR, no para predecir sobre datos nuevos."""
        horizonte = horizonte or config.obtener("horizonte_dias")

        prep = self.seleccionar_variables(df, columnas_clima)
        prep = self.tratar_valores_faltantes(prep, columnas_clima)
        prep = self.crear_variables_calendario(prep)
        prep = self.crear_variables_lag_y_objetivo(prep, horizonte)

        target_col = f"target_PotTotal_{horizonte}d"
        return prep.dropna(subset=["lag_14", target_col]).reset_index(drop=True)

    def preparar_dataset_prediccion(self, df: pd.DataFrame, columnas_clima: list,
                                     horizonte: int = None) -> pd.DataFrame:
        """
        Igual que preparar_dataset_modelado, pero NO exige que el target sea
        conocido (porque en producción, justamente, todavía no lo es). Solo
        descarta filas sin historial suficiente (lag_14). Las últimas filas
        del resultado son las que sirven para predecir el futuro real.
        """
        horizonte = horizonte or config.obtener("horizonte_dias")

        prep = self.seleccionar_variables(df, columnas_clima)
        prep = self.tratar_valores_faltantes(prep, columnas_clima)
        prep = self.crear_variables_calendario(prep)
        prep = self.crear_variables_lag_y_objetivo(prep, horizonte)

        return prep.dropna(subset=["lag_14"]).reset_index(drop=True)
