import os
#from dotenv import load_dotenv
import streamlit as st# pip install streamlit-autorefresh
import numpy as np
from PIL import Image
import time
import pandas as pd
from pathlib import Path
from streamlit_autorefresh import st_autorefresh
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

#load_dotenv()

#MONGODB_USER = os.getenv("MONGODB_USER")
#MONGODB_PASSWORD = os.getenv("MONGODB_PASSWORD")

if "kpi" not in st.session_state :
    st.session_state.kpi = 0

if "page" not in st.session_state :
    st.session_state.page = "Desktop" 

MODEL_PATH = f'./model/best_random_forest.pkl'
SCALER_PATH = f'./model/scaler.pkl'

df = pd.read_csv("./data/petrochemical_advanced_data.csv")
data = pd.read_csv("./data/motores_fallo_20_filas.csv")
data_2 = pd.read_csv("./data/motores_fallo_extra_1.csv")
data_3 = pd.read_csv("./data/motores_fallo_extra_2.csv")

def generar_dataset_temporal():
    muestra = df.sample(n=10)
    return muestra

def generar_dataset_temporal_2():
    muestra = df.sample(n=1)
    return muestra

#Function to load the Model and the Scaler
def load_pkl(fname):
    with open(fname, 'rb') as f:
        obj = pickle.load(f)
    return obj

model = load_pkl(MODEL_PATH)
#scaler = load_pkl(SCALER_PATH)


# Parse timestamp and extract time features
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df['Hour']  = df['Timestamp'].dt.hour    # 0-23 — time of day
df['Month'] = df['Timestamp'].dt.month   # 1-12 — seasonality

# Encode categorical columns (text → numbers)
le = LabelEncoder()
for col in ['Unit_Name', 'Catalyst_Type']:
    df[col] = le.fit_transform(df[col])

# Define features (X) and target (y)
drop_cols = ['Timestamp', 'Sensor_Health_Index']
features = [c for c in df.columns if c not in drop_cols]

X = df[features]
y = df['Sensor_Health_Index']

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

numerical_features = X.select_dtypes(include=['float64', 'int64'])

numerical_columns = numerical_features.columns

ct = ColumnTransformer([("only numeric", StandardScaler(), numerical_columns)], remainder='passthrough')

_ = ct.fit_transform(X_train)

def predict_kpi(input_data):
    kpi_values = pd.DataFrame([input_data], columns=features)

    # Scale the input data using the loaded scaler
    scaled_data = ct.transform(kpi_values)

    # Make predictions using the loaded model
    predictions = model.predict(scaled_data)

    return predictions

def main():
    st.set_page_config(
        page_title="Dashboard Motores Planta Turmero",
        page_icon="🛠️",
        layout="wide",
    )

    with st.sidebar:     
        columns = st.columns([0.1, 1, 0.1])

        with columns[1].container(border=False, height="stretch", vertical_alignment="center", gap="small", horizontal_alignment="center"):
            cols = st.columns([0.1, 1, 0.1])

            cols[1].image("./imgs/PE-0028_P.A.N._-Web-Peru_Bodegon_V5.png", width="stretch")
        st.sidebar.caption("🧑🏻‍💻 Revision de la Etiqueta del Empaque de un 1 Kilogramo de Harina P.A.N.", text_alignment="center")
        st.info("**Revision de los Paquetes de Harina P.A.N. de A.P.C Planta Turmero ↓**", icon="✅")
    
        with st.expander("🏭 Menu de Opciones", expanded=True):
            opciones = ("Desktop", "Mobile")
            st.session_state.page = st.selectbox("Vista:", opciones, index=1)

            st.caption("🧑🏻‍💻 Chequeo a Tiempo Real de los Parametros de los Motores de Planta Turmero.")
            st.success("Apoyado con el Modelo de A.I. de Computer Vision YOLOv11", icon="✅")

    #st_autorefresh(interval=7000, key="scada_refresh")

    st.title("⚒️ Parametros de los Sensores de los Equipos Planta 1 - Planta 2 de Harina", text_alignment="center")
    st.write(" ")
    columns = st.columns([0.3, 1, 0.3])

    with columns[1].container(border=False, height="stretch", vertical_alignment="center", gap="small", horizontal_alignment="center"):
        cols = st.columns([0.1, 1, 0.1])

        cols[1].image("./imgs/channels4_banner.jpg", width="stretch", caption="Empresas Polar - A.P.C Planta Turmero")
    
    
    st.header("🏭 Alimentos Polar Comercial Planta Turmero C.A. - Dashboard Prediction", text_alignment="center")
    st.write(" ")


    if st.session_state.page == "Desktop":
        st_autorefresh(interval=9000, key="scada_refresh")
        with st.expander("📑 EMPRESAS POLAR - A.P.C Planta Turmero", expanded=True):
            columns = st.columns([0.2, 1, 0.2])
            columns[1].caption("🧑🏻‍💻 Chequeo a Tiempo Real de los Parametros de los Motores de Planta Turmero.")
            columns[1].subheader("🏭 Comprobacion de Parametros como la Temperatura (°K), Revoluciones por Minuto (RPM) y Torque (Nm).")
            columns[1].caption("Con los presentes Parametros de los Sensores de los Motores de Planta Turmero, se Puede Realizar un Chequeo a Tiempo Real del Estado de los Motores, lo que Permite Detectar Posibles Anomalías o Desviaciones en su Funcionamiento, lo que Facilita la Toma de Decisiones para el Mantenimiento Preventivo y Correctivo en Planta Turmero.")
            columns[1].write(" ")
            columns[1].write(" ")
            with columns[1].container(border=False, height="stretch", vertical_alignment="center", gap="small", horizontal_alignment="center"):
                cols = st.columns([0.4, 1, 0.4])

                cols[1].image("./imgs/motor-electrico.jpg", width="stretch", caption="Motores Electricos en A.P.C Planta Turmero")
            columns[1].write(" ")
            columns[1].header("📊 Parametros de los Sensores de los Motores de Planta Turmero", text_alignment="center")
            columns[1].write(" ")
            with columns[1].container(border=True, height="stretch", vertical_alignment="center", gap="small"):
                
                cols = st.columns([1, 0.6], gap="medium")
                cols[0].write(" ")
                cols[0].subheader("📊 Parametros de los Sensores de los Motores de Planta Turmero", text_alignment="center")
                cols[0].write(" ")
                cols[0].dataframe(generar_dataset_temporal())
                
                cols[1].write(" ")
                cols[1].subheader("📊 Parametros de los Sensores de los Motores de Planta Turmero", text_alignment="center")
                cols[1].write(" ")
                with cols[1].container():
                    cols_6 = st.columns([0.3, 0.3], gap="medium")
                    vibration_level = generar_dataset_temporal_2()["Vibration_Level_mm_s"].values[0]
                    valve_opening_percentage = generar_dataset_temporal_2()["Valve_Opening_Percent"].values[0]
                    feedstock_flow = generar_dataset_temporal_2()["Feedstock_Flow_m3h"].values[0]
                    electricity_mwh = generar_dataset_temporal_2()["Electricity_MWh"].values[0]
                    reactor_temp_c = generar_dataset_temporal_2()["Reactor_Temp_C"].values[0]
                    reactor_pressure_bar = generar_dataset_temporal_2()["Reactor_Pressure_Bar"].values[0]

                    with cols_6[0].container(border=True):
                        
                        st.metric(
                            "Vibration Level (mm/s)",
                            f"{round(vibration_level, 2)} (mm/s)",
                            delta="MOTOR ON",
                        )

                    with cols_6[0].container(border=True):
                        st.metric(
                            "Valve Opening Percentage (%)",
                            f"{round(valve_opening_percentage, 2)} %",
                            delta="MOTOR ON",
                        )
                    with cols_6[0].container(border=True):
                        st.metric(
                            "Feedstock Flow (m³/h)",
                            f"{round(feedstock_flow, 2)} m³/h",
                            delta="MOTOR ON",
                        )

                    with cols_6[1].container(border=True):
                        st.metric(
                            "Electricity Consumption (MWh)",
                            f"{round(electricity_mwh, 2)} MWh",
                            delta="MOTOR ON",
                        )
                    with cols_6[1].container(border=True):
                        st.metric(
                            "Reactor Temperature (°C)",
                            f"{round(reactor_temp_c, 2)} °C",
                            delta="MOTOR ON",
                        )
                    with cols_6[1].container(border=True):
                        st.metric(
                            "Reactor Pressure (Bar)",
                            f"{round(reactor_pressure_bar, 2)} Bar",
                            delta="MOTOR ON",
                        )
                cols[1].write(" ")
                
            #columns[1].info("**Selecciona y Suba la Imagen del Paquete de 1 Kilogramo de Harina P.A.N. para Comprobar su Etiqueta**", icon="ℹ️")
            #columns[1].divider()
            columns[1].write(" ")
            
            with columns[1].container(border=True, height="stretch", vertical_alignment="center", gap="small"):
                cols = st.columns([1, 1], gap="medium")
                cols[0].write(" ")  
                cols[0].write(" ") 
                with cols[0].container(border=False, height="stretch", vertical_alignment="center", gap="small", horizontal_alignment="center"):
                    cols_3 = st.columns([0.2, 1, 0.2])
                    cols_3[1].image("./imgs/bodegon (1).png", width="stretch", caption="Linea de Productos de Empresas Polar - A.P.C Planta Turmero")
                
                cols[0].divider()
                cols[0].header("📊 Parametros de los Sensores de los Motores de Planta Turmero", text_alignment="center")
                cols[0].write(" ")
                with cols[0].container(border=False):
                    st.info("Resultado de la Comprobacion de la Etiqueta del Paquete de 1 Kilogramo de Harina P.A.N.", icon="✅")
                    st.divider()
                    cols_2 = st.columns([1, 1], gap="small")
                    unit = cols_2[0].selectbox("Unit", ("Ethylene_Plant_01", "Ammonia_Unit_02", "Methanol_Complex_03"), key="selectbox1")
                    catalyst_age_days = cols_2[0].slider("Catalyst Age Days", 0, 365, 100, key="slider8")
                    vibration_level_mm_s = cols_2[0].slider("Vibration Level (mm/s)", 0.5, 8.5, 4.0, key="slider10")
                    reactor_temp_c = cols_2[0].slider("Reactor Temperature (°C)", 670.0, 980.0, 700.0, key="slider11")
                    reactor_pressure_bar = cols_2[0].slider("Reactor Pressure (Bar)", 20.0, 45.0, 30.0, key="slider13")
                    steam_tons_h = cols_2[0].slider("Steam (Tons/H)", 40.0, 95.0, 60.0, key="slider17")
                    product_yield_tons = cols_2[0].slider("Product Yield (Tons)", 40.0, 135.0, 60.0, key="slider19")
                    hour = cols_2[0].slider("Hour", 0, 24, 12, key="slider21")

                    catalyst_type = cols_2[1].selectbox("Catalyst Type", ("Platinum_Base_X1", "Cobalt_Premium_Z2", "Iron_Standard_V5"), key="selectbox2")
                    valve_opening_percent = cols_2[1].slider("Valve Opening Percent (%)", 0.0, 100.0, 50.0, key="slider14")
                    feedstock_flow_m3h = cols_2[1].slider("Feedstock Flow (M3h)", 341.0, 760.0, 300.0, step=0.1, key="slider12")
                    electricity_mwh = cols_2[1].slider("Electricity (MWh)", 15.0, 30.0, 20.0, step=0.1, key="slider15")
                    natural_gas_m3h = cols_2[1].slider("Natural Gas (M3h)", 3000.0, 5200.0, 4000.0, step=0.1, key="slider16")
                    ambient_temperature_c = cols_2[1].slider("Ambient Temperature (°C)", -10.0, 52.0, 20.0, step=0.1, key="slider18")
                    energy_intensity = cols_2[1].slider("Energy Intensity", 1.0, 7.0, 3.0, step=0.1, key="slider20")
                    month = cols_2[1].slider("Month", 1, 12, 6, key="slider22")

                    if unit == "Ethylene_Plant_01":
                        unit = 1
                    elif unit == "Ammonia_Unit_02":
                        unit = 0
                    elif unit == "Methanol_Complex_03":
                        unit = 2

                    if catalyst_type == "Platinum_Base_X1":
                        catalyst_type = 2
                    elif catalyst_type == "Cobalt_Premium_Z2":
                        catalyst_type = 0
                    elif catalyst_type == "Iron_Standard_V5":
                        catalyst_type = 1

                    data_kpi = [unit, catalyst_type, catalyst_age_days, vibration_level_mm_s, valve_opening_percent, feedstock_flow_m3h, reactor_temp_c, reactor_pressure_bar, electricity_mwh, natural_gas_m3h, steam_tons_h, ambient_temperature_c, product_yield_tons, energy_intensity, hour, month]

                    st.write(" ")
                    if st.button("Comprobar Etiqueta del Paquete de 1 Kilogramo de Harina P.A.N.", type="primary", use_container_width=True) :
                        st.session_state.kpi = predict_kpi(data_kpi)
                    st.write(" ")
                    

                cols[1].write(" ")
                cols[1].write(" ")
                cols[1].header("📊 Parametros de los Sensores de los Motores de Planta Turmero", text_alignment="center")
                cols[1].write(" ")
                cols[1].success("Resultado de la Comprobacion de la Etiqueta del Paquete de 1 Kilogramo de Harina P.A.N.", icon="✅")
                cols[1].divider()

                with cols[1].container(border=False, height="stretch", vertical_alignment="center", gap="small", horizontal_alignment="center"):
                    #st.header("📊 Parametros de los Sensores de los Motores de Planta Turmero", text_alignment="center")
                    st.write(" ")
                    cols_3 = st.columns([0.2, 1, 0.2])
                    cols_3[1].image("./imgs/motores-trifasicos-promelsa.jpg", width="stretch", caption="Linea de Productos de Empresas Polar - A.P.C Planta Turmero")
                cols[1].divider()
                cols[1].metric(
                    "RESULTADO PREDECTIVO DEL KPI DEL EQUIPO",
                    f"{round(st.session_state.kpi[0], 2)} KPI - {round(st.session_state.kpi[0], 2) * 100} %",
                    delta=(f"VALOR DEL KPI DE {round(st.session_state.kpi[0], 2)} - {round(st.session_state.kpi[0], 2) * 100} %"),
                )

                cols[1].caption("El Modelo de Computer Vision YOLOv11 ha Detectado que la Etiqueta del Paquete de 1 Kilogramo de Harina P.A.N. se Encuentra Impresa Correctamente, lo que Indica un Proceso de Impresion de la Etiqueta sin Errores en Empaque de Planta Turmero.")
                
                cols[1].divider()
                with cols[1].container(border=False, height="stretch", vertical_alignment="center", gap="small", horizontal_alignment="center"):
                    if round(st.session_state.kpi[0], 2) > 0.5 :
                        st.header(f"📊 VALOR OPTIMO DEL KPI - {round(st.session_state.kpi[0], 2)}", text_alignment="center")
                        st.write(" ")
                        st.write(" ")
                        cols_4 = st.columns([0.2, 1, 0.2])
                        cols_4[1].image("./imgs/indicador_transparente_sin_fondo.png", width="stretch")
                        #cols_4[1].write(" ")
                        with cols_4[1].container(border=False, height="stretch", vertical_alignment="center", gap="small", horizontal_alignment="center"):
                            cols_5 = st.columns([0.45, 1, 0.2])
                            cols_5[1].caption("Prediccion del Valor del KPI del Equipo")
                    elif round(st.session_state.kpi[0], 2) <= 0.5 :
                        st.header(f"📊 VALOR DEFICIENTE DEL KPI - {round(st.session_state.kpi[0], 2)}", text_alignment="center")
                        st.write(" ")
                        cols_4 = st.columns([0.2, 1, 0.2])
                        cols_4[1].image("./imgs/indicador_rojo_misma_imagen_fondo_transparente.png", width="stretch")
                        #cols_4[1].write(" ")
                        with cols_4[1].container(border=False, height="stretch", vertical_alignment="center", gap="small", horizontal_alignment="center"):
                            cols_5 = st.columns([0.45, 1, 0.2])
                            cols_5[1].caption("Prediccion del Valor del KPI del Equipo")
                

                #if cols[1].button("Verificar otra Etiqueta de un Paquete de Harina P.A.N.", type="primary", use_container_width=True, on_click=lambda: st.session_state.update({"paquete": None, "percentage": None})) :
                    #st.rerun()
            columns[1].write(" ")
            columns[1].write(" ")


    elif st.session_state.page == "Mobile":
        with st.expander("📑 EMPRESAS POLAR - A.P.C Planta Turmero", expanded=True):
            columns = st.columns([0.2, 1, 0.2])
            columns[1].caption("🧑🏻‍💻 Revision de la Etiqueta del Empaque de un 1 Kilogramo de Harina P.A.N.")
            columns[1].subheader("🏭 Comprobacion de Impresion Correcta de la Etiqueta Utilizando Modelo de Computer Vision YOLOv11")
            columns[1].caption("Este modelo de Computer Vision YOLOv11 ha sido Entrenado para Detectar la Presencia o Ausencia de la Etiqueta en el Empaque de un Paquete de 1 Kilogramo de Harina P.A.N., lo que Permite Verificar si la Etiqueta se Encuentra Impresa Correctamente en el Proceso de Empaque en Planta Turmero.")

            columns[1].divider()
            columns[1].info("**Selecciona y Suba la Imagen del Paquete de 1 Kilogramo de Harina P.A.N. para Comprobar su Etiqueta**", icon="ℹ️")

            with columns[1].container(border=True, height="stretch", vertical_alignment="center", gap="small", horizontal_alignment="center"):
                st.write(" ")
                st.success("Etiqueta Correcta del Paquete de 1 Kilogramo de Harina P.A.N.", icon="✅")
                st.write(" ")
                st.image("./imgs/142541 - Copy_jpg.rf.dT2SivX7muzgKYnXNJoX.jpg", width=250, caption="Paquete con su Etiqueta de 1 Kilogramo de Harina P.A.N. - A.P.C Planta Turmero")
                
                st.divider()
                st.caption("Este modelo de Computer Vision YOLOv11 ha sido Entrenado para Detectar la Presencia o Ausencia de la Etiqueta en el Empaque de un Paquete de 1 Kilogramo de Harina P.A.N., lo que Permite Verificar si la Etiqueta se Encuentra Impresa Correctamente en el Proceso de Empaque en Planta Turmero.")
                st.image("./imgs/bodegon (1).png", width="stretch", caption="Linea de Productos de Empresas Polar - A.P.C Planta Turmero")
                st.divider()
                
                st.write(" ")
                st.warning("Etiqueta Incorrecta del Paquete de 1 Kilogramo de Harina P.A.N.", icon="⚠️")
                st.write(" ")
                st.image("./imgs/Harina-A-Base-De-Maiz-Pan-Precocido.jpg", width=250, caption="Paquete sin Etiqueta de 1 Kilogramo de Harina P.A.N. - A.P.C Planta Turmero")
                
            img = columns[1].file_uploader("Carga Foto del Paquete de 1 Kg de Harina P.A.N.: ", type=["png", "jpg", "jpeg"])

            if img is not None:
                with columns[1].container(border=True, height="stretch", vertical_alignment="center", gap="small", horizontal_alignment="center"):
                    image = np.array(Image.open(img))   
                    st.write("")
                    st.success("Etiqueta Correcta del Paquete de 1 Kilogramo de Harina P.A.N.", icon="✅")
                    st.divider()
                    st.image("./testing-imgs/etiqueta/Project-17-63 - Copy_jpg.rf.puWf1Z83bfqaGm1GwE23.jpg", width=300, caption="Paquete de 1 Kilogramo de Harina P.A.N. - A.P.C Planta Turmero")
                    st.write("")
                    st.write("")
                    st.error("Etiqueta Incorrecta del Paquete de 1 Kilogramo de Harina P.A.N.", icon="❌")
                    st.divider()
                    st.image("./testing-imgs/no_etiqueta/Group-371_png.rf.h0TvHKLrStMIrAlUE1uQ.png", width=300, caption="Paquete de 1 Kilogramo de Harina P.A.N. - A.P.C Planta Turmero")
                    st.info("Imagen del Paquete de Harina de 1 Kilogramo P.A.N. Cargado para Verificar.", icon="✅")
                    st.divider()
                    st.image(image, width=400, caption="Paquete de 1 Kilogramo de Harina P.A.N. - A.P.C Planta Turmero")
                    st.divider()
                    if st.button("Comprobar Etiqueta del Paquete de 1 Kilogramo de Harina P.A.N.", type="primary", use_container_width=True) :
                        pass

            if st.session_state.paquete is not None and st.session_state.percentage is not None:
                with columns[1].container(border=True, height="stretch", vertical_alignment="center", gap="small"):
                    st.image("./imgs/PE-0028_P.A.N._-Web-Peru_Bodegon_V5.png", width="stretch", caption="Linea de Productos de Empresas Polar - A.P.C Planta Turmero")

                    if st.session_state.paquete == 0 :
                        st.write(" ")
                        my_bar = st.progress(0, text=" ")
                        for percent_complete in range(100):
                            time.sleep(0.01)
                            my_bar.progress(percent_complete + 1, text=" ")
                        time.sleep(1)
                        my_bar.empty()

                        st.divider()
                        st.success("Resultado de la Comprobacion de la Etiqueta del Paquete de 1 Kilogramo de Harina P.A.N.", icon="✅")
                        st.write(" ")
                        st.metric(
                            "RESULTADO DE LA COMPROBACION DE LA ETIQUETA",
                            "¡Etiqueta Impresa Correctamente!",
                            delta=(f"{st.session_state.percentage} %"),
                        )
                        st.caption("El Modelo de Computer Vision YOLOv11 ha Detectado que la Etiqueta del Paquete de 1 Kilogramo de Harina P.A.N. se Encuentra Impresa Correctamente, lo que Indica un Proceso de Impresion de la Etiqueta sin Errores en Empaque de Planta Turmero.")
                        st.divider()
                    elif st.session_state.paquete == 1 :
                        st.write(" ")
                        my_bar = st.progress(0, text=" ")
                        for percent_complete in range(100):
                            time.sleep(0.01)
                            my_bar.progress(percent_complete + 1, text=" ")
                        time.sleep(1)
                        my_bar.empty()

                        st.divider()
                        st.error("Resultado de la Comprobacion de la Etiqueta del Paquete de 1 Kilogramo de Harina P.A.N.", icon="❌")
                        st.write(" ")
                        st.metric(
                            "RESULTADO DE LA COMPROBACION DE LA ETIQUETA: ",
                            "¡Etiqueta No Impresa Correctamente!",
                            delta=(f"Accuracy del {st.session_state.percentage} %"),
                        )
                        st.caption("El Modelo de Computer Vision YOLOv11 ha Detectado que la Etiqueta del Paquete de 1 Kilogramo de Harina P.A.N. no se Encuentra Impresa Correctamente, lo que Indica un Posible Error en el Proceso de Impresion de la Etiqueta en Empaque de Planta Turmero.")
                        st.divider()

                    if st.button("Verificar otra Etiqueta de un Paquete de Harina P.A.N.", type="primary", use_container_width=True, on_click=lambda: st.session_state.update({"paquete": None, "percentage": None})) :
                        st.rerun()
            
if __name__ == "__main__":
    main()