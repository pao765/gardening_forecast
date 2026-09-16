
"""
UBICACIÓN: prediction_potTotal/modeling/train.py

Corre el pipeline completo, identifica el experimento ganador por
MAE, lo re-entrena y guarda el modelo final + preprocesador.
Además, registra cada experimento en MLflow.

Se ejecuta con: python -m prediction_potTotal.modeling.train
"""

import json
import joblib
import numpy as np
import mlflow
import mlflow.sklearn

from prediction_potTotal.config import MODELS_DIR
from prediction_potTotal.features import Preprocesador
from prediction_potTotal.modeling.models import ModeloRandomForest, ModeloXGBoost
from prediction_potTotal.pipeline import COLUMNAS_CLIMA, Pipeline


def main():
    # ===== CONFIGURAR MLFLOW =====
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Prediccion_PotTotal_Horticola")
    
    pipeline = Pipeline()
    tabla = pipeline.comparar_todo()
    mejor = tabla.iloc[0]
    print(f"Mejor experimento: {mejor['Modelo']} (MAE={mejor['MAE']})")

    train, test = pipeline.preparar_datos()
    y_train_raw = train[pipeline.target_col].values

    cols_lags = ["lag_7", "lag_14", "rolling_mean_7", "rolling_mean_14"]
    usa_clima = "con clima" in mejor["Configuración"]
    cols_numericas = COLUMNAS_CLIMA + cols_lags if usa_clima else cols_lags

    preprocesador_final = Preprocesador()
    preprocesador_final.ajustar_transformadores(train, cols_numericas)
    X_train = preprocesador_final.transformar(train)

    usa_log = mejor["Target"] == "log"
    y_final = np.log1p(y_train_raw) if usa_log else y_train_raw

    modelo_final = (
        ModeloRandomForest() if mejor["Algoritmo"] == "Random Forest" else ModeloXGBoost()
    )

    # ===== REGISTRAR EN MLFLOW =====
    with mlflow.start_run(run_name=f"Modelo_Final_{mejor['Algoritmo']}") as run:
        # 1. Tags
        mlflow.set_tags({
            "proyecto": "Prediccion_PotTotal_Horticola",
            "horizonte": f"{pipeline.horizonte}d",
            "framework": "scikit-learn/xgboost"
        })
        
        # 2. Parámetros
        mlflow.log_param("algoritmo", mejor["Algoritmo"])
        mlflow.log_param("usa_clima", usa_clima)
        mlflow.log_param("usa_log", usa_log)
        mlflow.log_param("horizonte_dias", pipeline.horizonte)
        mlflow.log_param("n_features", len(cols_numericas))
        mlflow.log_param("configuracion", mejor["Configuración"])
        
        # 3. Entrenar
        modelo_final.entrenar(X_train, y_final)
        
        # 4. Métricas
        mlflow.log_metric("mae_validacion", float(mejor["MAE"]))
        
        # 5. Registrar modelo (con manejo de atributo .modelo)
        try:
            mlflow.sklearn.log_model(
                sk_model=modelo_final.modelo,
                artifact_path="model",
                registered_model_name="PotTotal_Horticola_Model"
            )
        except AttributeError:
            # Si no tiene atributo .modelo, guardar el objeto completo
            mlflow.sklearn.log_model(
                sk_model=modelo_final,
                artifact_path="model",
                registered_model_name="PotTotal_Horticola_Model"
            )
        
        # 6. Metadata como artefacto
        metadata = {
            "algoritmo": mejor["Algoritmo"],
            "usa_clima": usa_clima,
            "usa_log": usa_log,
            "columnas_numericas": cols_numericas,
            "horizonte_dias": pipeline.horizonte,
            "mae_validacion": float(mejor["MAE"]),
        }
        mlflow.log_dict(metadata, "metadata_modelo.json")
        
        print(f"[MLflow] Run ID: {run.info.run_id}")
    # ===== FIN MLFLOW =====

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    modelo_final.guardar_modelo(MODELS_DIR / "modelo_pot_total_7d.pkl")
    joblib.dump(preprocesador_final, MODELS_DIR / "preprocesador_pot_total_7d.pkl")

    with open(MODELS_DIR / "metadata_modelo.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"Modelo guardado en {MODELS_DIR / 'modelo_pot_total_7d.pkl'}")
    print(f"Preprocesador guardado en {MODELS_DIR / 'preprocesador_pot_total_7d.pkl'}")
    print(f"Metadata guardada en {MODELS_DIR / 'metadata_modelo.json'}")


if __name__ == "__main__":
    main()