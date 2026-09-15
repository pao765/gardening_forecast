"""
UBICACIÓN: prediction_potTotal/eda.py

Clase AnalisisExploratorio: agrupa todos los análisis que ya hiciste
en el notebook (calidad de datos, distribución de PotTotal,
estacionalidad, outliers, relación con clima y calendario).
Se usa desde un notebook de EDA, pasándole el DataFrame ya cargado.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


class AnalisisExploratorio:
    """EDA para el dataset de ventas del comercio de horticultura."""

    # --- Estructura y calidad de datos ---

    def resumen_general(self, df: pd.DataFrame) -> None:
        print(f"Dimensiones: {df.shape[0]} filas x {df.shape[1]} columnas")
        print(f"Fecha mínima: {df['Date'].min().date()}")
        print(f"Fecha máxima: {df['Date'].max().date()}")

    def verificar_continuidad_temporal(self, df: pd.DataFrame) -> pd.DatetimeIndex:
        """Detecta huecos en la secuencia de fechas (días faltantes)."""
        rango_completo = pd.date_range(df["Date"].min(), df["Date"].max(), freq="D")
        dias_faltantes = rango_completo.difference(df["Date"])
        print(f"Días esperados en el rango: {len(rango_completo)}")
        print(f"Días faltantes en la serie: {len(dias_faltantes)}")
        return dias_faltantes

    def resumen_nulos(self, df: pd.DataFrame) -> pd.DataFrame:
        nulos = df.isna().sum()
        porcentaje = (df.isna().mean() * 100).round(2)
        tabla = pd.DataFrame({"nulos": nulos, "porcentaje_%": porcentaje})
        tabla = tabla[tabla["nulos"] > 0].sort_values("porcentaje_%", ascending=False)
        print(tabla if not tabla.empty else "Sin valores nulos.")
        return tabla

    def detectar_duplicados(self, df: pd.DataFrame) -> None:
        print("Filas totalmente duplicadas:", df.duplicated().sum())
        print("Fechas duplicadas:", df["Date"].duplicated().sum())

    def detectar_valores_negativos(self, df: pd.DataFrame, columnas_venta: list) -> dict:
        negativos = {c: int((df[c] < 0).sum()) for c in columnas_venta}
        print("Valores negativos por columna de venta:", negativos)
        return negativos

    def resumen_estadistico(self, df: pd.DataFrame, columnas: list) -> pd.DataFrame:
        descripcion = df[columnas].describe().round(2)
        print(descripcion)
        return descripcion

    # --- Análisis de la variable objetivo ---

    def analizar_pot_total(self, df: pd.DataFrame) -> None:
        print(df["PotTotal"].describe().round(2))
        print(f"Asimetría (skewness): {df['PotTotal'].skew():.2f}")
        print(f"Curtosis: {df['PotTotal'].kurt():.2f}")
        print(f"Días con venta = 0: {(df['PotTotal'] == 0).sum()} "
              f"({(df['PotTotal'] == 0).mean() * 100:.1f}%)")

    def graficar_distribucion_pot_total(self, df: pd.DataFrame) -> None:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        axes[0].hist(df["PotTotal"], bins=40, color="#55A868", edgecolor="white")
        axes[0].axvline(df["PotTotal"].mean(), color="black", linestyle="--",
                         label=f"Media: {df['PotTotal'].mean():.1f}")
        axes[0].axvline(df["PotTotal"].median(), color="red", linestyle=":",
                         label=f"Mediana: {df['PotTotal'].median():.1f}")
        axes[0].set_title("Distribución de PotTotal")
        axes[0].legend()

        axes[1].hist(np.log1p(df["PotTotal"]), bins=40, color="#8172B2", edgecolor="white")
        axes[1].set_title("Distribución de log(1 + PotTotal)")
        plt.tight_layout()
        plt.show()

    def graficar_serie_tiempo(self, df: pd.DataFrame, columna: str = "PotTotal") -> None:
        fig, ax = plt.subplots(figsize=(13, 4.5))
        ax.plot(df["Date"], df[columna], color="#55A868", linewidth=0.8, alpha=0.9)
        ax.set_title(f"Serie de tiempo de {columna}")
        plt.tight_layout()
        plt.show()

    def graficar_componentes_pot_total(self, df: pd.DataFrame) -> None:
        fig, ax = plt.subplots(figsize=(13, 4.5))
        ax.plot(df["Date"], df["PotOwn"], label="PotOwn", linewidth=0.8, alpha=0.8)
        ax.plot(df["Date"], df["PotPurchased"], label="PotPurchased", linewidth=0.8, alpha=0.8)
        ax.set_title("Componentes de PotTotal: producción propia vs. compradas")
        ax.legend()
        plt.tight_layout()
        plt.show()

    # --- Variables climáticas ---

    def graficar_distribucion_clima(self, df: pd.DataFrame, columnas_clima: list) -> None:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        for ax, col in zip(axes.flat, columnas_clima):
            ax.hist(df[col].dropna(), bins=35, color="#8172B2", edgecolor="white")
            ax.set_title(col)
        plt.tight_layout()
        plt.show()

    def graficar_matriz_correlacion(self, df: pd.DataFrame, columnas: list) -> None:
        corr = df[columnas].corr()
        plt.figure(figsize=(8, 6))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True)
        plt.title("Matriz de correlación")
        plt.tight_layout()
        plt.show()

    # --- Calendario ---

    def graficar_ventas_por_dia_semana(self, df: pd.DataFrame) -> pd.DataFrame:
        orden_dias = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        resumen = df.groupby("weekday")[["PotTotal"]].mean().reindex(orden_dias).round(1)
        resumen.plot(kind="bar", figsize=(9, 4.5), color="#55A868", legend=False)
        plt.title("PotTotal promedio por día de la semana")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        return resumen

    def graficar_estacionalidad_mensual(self, df: pd.DataFrame) -> pd.DataFrame:
        resumen = df.groupby("month")[["PotTotal"]].mean().round(1)
        resumen.plot(kind="line", marker="o", figsize=(9, 4.5), color="#55A868", legend=False)
        plt.title("Estacionalidad mensual de PotTotal")
        plt.xticks(range(1, 13))
        plt.tight_layout()
        plt.show()
        return resumen

    # --- Outliers ---

    def detectar_outliers_iqr(self, df: pd.DataFrame, columna: str = "PotTotal") -> dict:
        q1, q3 = df[columna].quantile([0.25, 0.75])
        iqr = q3 - q1
        lim_inf, lim_sup = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outliers = ((df[columna] < lim_inf) | (df[columna] > lim_sup)).sum()
        resultado = {
            "Q1": round(q1, 1), "Q3": round(q3, 1), "IQR": round(iqr, 1),
            "limite_inferior": round(lim_inf, 1), "limite_superior": round(lim_sup, 1),
            "n_outliers": int(outliers),
            "pct_outliers": round(outliers / len(df) * 100, 2),
        }
        print(resultado)
        return resultado

    # --- Problemas específicos (inflación de ceros) ---

    def analizar_inflacion_ceros(self, df: pd.DataFrame) -> None:
        pct_cero = (df["PotTotal"] == 0).mean() * 100
        print(f"Días con PotTotal = 0: {pct_cero:.1f}%")

        dias_cero = df[df["PotTotal"] == 0]
        if "weekday" in df.columns:
            pct_domingo = (dias_cero["weekday"] == "Sunday").mean() * 100
            print(f"De esos días, {pct_domingo:.0f}% son domingos.")
