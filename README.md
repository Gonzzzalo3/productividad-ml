# Predicción de productividad en teletrabajo

Este proyecto predice la productividad de empleados (tareas/semana) a partir de variables como satisfacción con el equipo, horas de formación, estrés, jornada continua, etc.

## Estructura
- EDA completo con visualizaciones.
- ETL con transformaciones logarítmicas, codificación one‑hot y escalado.
- Modelado con XGBoost (mejor modelo tras ajuste de hiperparámetros).
- Métricas finales: RMSE = 2.82, MAE = 2.29, R² = 0.634.
- Despliegue en Streamlit Cloud (app interactiva).

## Tecnologías
- Python (pandas, numpy, scikit-learn, xgboost, joblib)
- Streamlit (frontend)
- Docker (opcional)
- GitHub + Streamlit Cloud

## Uso
Clona el repositorio, instala dependencias (`pip install -r requirements.txt`) y ejecuta `streamlit run app.py`.
