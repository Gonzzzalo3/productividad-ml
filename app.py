import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Configuración de la página
st.set_page_config(page_title="Predicción de Productividad", layout="centered")
st.title("📊 Predicción de Productividad de Empleados")
st.markdown("Ingresa los datos del empleado y obtén la productividad estimada (tareas/semana).")

# Cargar modelo y escalador (con caché para eficiencia)
@st.cache_resource
def load_model():
    modelo = joblib.load('models/xgboost_final.pkl')
    scaler = joblib.load('models/scaler.pkl')
    return modelo, scaler

modelo, scaler = load_model()

# Lista de columnas que usó el modelo (mismo orden que en entrenamiento)
columnas_entrenamiento = [
    'edad', 'años_empresa', 'satisfaccion_equipo', 'estres_reportado',
    'horas_formacion_mes_log', 'distancia_oficina_km_log', 'bono_anual_log',
    'hijos_bin', 'jornada_bin',
    'sexo_Mujer', 'sexo_No binario',
    'departamento_Marketing', 'departamento_Operaciones',
    'departamento_RRHH', 'departamento_Ventas'
]

# Columnas numéricas a escalar (continuas)
columnas_a_escalar = [
    'edad', 'años_empresa', 'satisfaccion_equipo', 'estres_reportado',
    'horas_formacion_mes_log', 'distancia_oficina_km_log', 'bono_anual_log'
]

# --- Formulario de entrada ---
with st.form("prediccion_form"):
    st.subheader("Datos del empleado")

    # Variables numéricas originales
    edad = st.number_input("Edad (años)", min_value=18, max_value=70, value=35)
    años_empresa = st.number_input("Años en la empresa", min_value=0, max_value=40, value=5)
    satisfaccion_equipo = st.slider("Satisfacción con el equipo (1-10)", 1, 10, 7)
    estres_reportado = st.slider("Nivel de estrés reportado (1-10)", 1, 10, 4)
    horas_formacion = st.number_input("Horas de formación al mes", min_value=0.0, max_value=50.0, value=12.0)
    distancia_oficina = st.number_input("Distancia a la oficina (km)", min_value=0.0, max_value=100.0, value=5.0)
    bono_anual = st.number_input("Bono anual (miles €)", min_value=0.0, max_value=30.0, value=4.0)

    # Variables categóricas
    hijos = st.selectbox("¿Tiene hijos?", ["No", "Sí"])
    jornada = st.selectbox("¿Jornada continua?", ["No", "Sí"])
    sexo = st.selectbox("Sexo", ["Hombre", "Mujer", "No binario"])
    departamento = st.selectbox("Departamento", ["IT", "Ventas", "Marketing", "Operaciones", "RRHH"])

    submitted = st.form_submit_button("Predecir productividad")

# --- Procesamiento y predicción ---
if submitted:
    # Construir DataFrame con una fila
    nuevo = pd.DataFrame([{
        'edad': edad,
        'años_empresa': años_empresa,
        'satisfaccion_equipo': satisfaccion_equipo,
        'estres_reportado': estres_reportado,
        'horas_formacion_mes': horas_formacion,
        'distancia_oficina_km': distancia_oficina,
        'bono_anual': bono_anual,
        'hijos': hijos,
        'jornada_continua': jornada,
        'sexo': sexo,
        'departamento': departamento
    }])

    # 1. Aplicar log1p a variables con sesgo
    nuevo['horas_formacion_mes_log'] = np.log1p(nuevo['horas_formacion_mes'])
    nuevo['distancia_oficina_km_log'] = np.log1p(nuevo['distancia_oficina_km'])
    nuevo['bono_anual_log'] = np.log1p(nuevo['bono_anual'])

    # 2. Eliminar columnas originales no transformadas
    nuevo = nuevo.drop(columns=['horas_formacion_mes', 'distancia_oficina_km', 'bono_anual'])

    # 3. Codificar variables binarias
    nuevo['hijos_bin'] = nuevo['hijos'].map({'No': 0, 'Sí': 1})
    nuevo['jornada_bin'] = nuevo['jornada_continua'].map({'No': 0, 'Sí': 1})

    # 4. One-hot encoding para sexo y departamento (drop_first=True)
    nuevo = pd.get_dummies(nuevo, columns=['sexo', 'departamento'], drop_first=True)

    # 5. Rellenar columnas dummy faltantes con 0 y ordenar
    for col in columnas_entrenamiento:
        if col not in nuevo.columns:
            nuevo[col] = 0
    nuevo = nuevo[columnas_entrenamiento]

    # 6. Escalar las columnas continuas
    nuevo[columnas_a_escalar] = scaler.transform(nuevo[columnas_a_escalar])

    # 7. Predecir
    prediccion = modelo.predict(nuevo)[0]

    st.success(f"📈 Productividad estimada: **{prediccion:.1f}** tareas por semana")