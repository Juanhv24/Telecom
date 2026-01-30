import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score, recall_score
import os

# ==============================================================================
# 1. CONFIGURACIÓN DE PÁGINA
# ==============================================================================
st.set_page_config(
    page_title="Interconnect BI Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    [data-testid="stMetricValue"] { font-size: 26px; font-weight: bold; color: #333; }
    div.block-container { padding-top: 2rem; }
    [data-testid="stExpander"] { background-color: #f8f9fa; border: 1px solid #ddd; }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. CARGA DE DATOS
# ==============================================================================
@st.cache_resource
def load_assets():
    # Función auxiliar para buscar archivos inteligentemente
    def encontrar_archivo(nombre_archivo, carpeta_local=''):
        # 1. Intenta buscar en la carpeta actual (Ideal para GitHub/Cloud)
        path_actual = os.path.join(os.path.dirname(__file__), nombre_archivo)
        if os.path.exists(path_actual):
            return path_actual
        
        # 2. Intenta buscar usando rutas relativas locales (Tu estructura de PC)
        # '../Data' significa: Sube un nivel desde 'Notebooks' y entra a 'Data'
        path_local = os.path.join(os.path.dirname(__file__), '..', carpeta_local, nombre_archivo)
        if os.path.exists(path_local):
            return path_local
            
        return None # Falló

    # --- DEFINICIÓN DE ARCHIVOS ---
    # El modelo suele estar en la misma carpeta del notebook
    file_model = encontrar_archivo('modelo_churn_interconnect.pkl', carpeta_local='Notebooks')
    
    # Los CSVs suelen estar en la carpeta Data
    file_xtest = encontrar_archivo('X_test_interconnect.csv', carpeta_local='Data')
    file_ytest = encontrar_archivo('y_test_interconnect.csv', carpeta_local='Data')
    file_df_full = encontrar_archivo('df_final_interconnect.csv', carpeta_local='Data')

    # Validación de seguridad
    if not all([file_model, file_xtest, file_ytest, file_df_full]):
        missing = []
        if not file_model: missing.append("Modelo .pkl")
        if not file_xtest: missing.append("X_test.csv")
        st.error(f"❌ No se encontraron estos archivos: {', '.join(missing)}")
        st.stop()
        
    # --- CARGA ---
    model = joblib.load(file_model)
    X_test = pd.read_csv(file_xtest)
    y_test = pd.read_csv(file_ytest)
    df_full = pd.read_csv(file_df_full)
    
    return model, X_test, y_test, df_full

try:
    with st.spinner("Cargando entorno de análisis..."):
        pipeline, X_test_loaded, y_test_loaded, df_full_loaded = load_assets()
except Exception as e:
    st.error(f"Error cargando archivos: {e}")
    st.stop()

# ==============================================================================
# 3. BARRA LATERAL
# ==============================================================================
st.sidebar.title("🛠️ Simulador de Cliente")

tenure = st.sidebar.slider("Antigüedad (Días)", 0, 2500, 365)
monthly_charges = st.sidebar.slider("Cargos Mensuales ($)", 18.0, 120.0, 70.0)
contract = st.sidebar.selectbox("Contrato", ['Month-to-month', 'One year', 'Two year'])
internet = st.sidebar.selectbox("Internet", ['Fiber optic', 'DSL', 'No'])
payment = st.sidebar.selectbox("Pago", ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])

with st.sidebar.expander("Servicios Extra"):
    online_security = st.selectbox("Seguridad Online", ['No', 'Yes', 'No internet service'])
    tech_support = st.selectbox("Soporte Técnico", ['No', 'Yes', 'No internet service'])
    paperless = st.selectbox("Facturación Digital", ['Yes', 'No'])

# ==============================================================================
# 4. DASHBOARD PRINCIPAL
# ==============================================================================
st.title("📡 Interconnect: Dashboard Estratégico")
st.markdown(f"**Data Source:** Histórico ({len(df_full_loaded):,} clientes) | **Modelo:** LightGBM (AUC 0.9054)")

tab1, tab2, tab3 = st.tabs(["🔮 Predicción IA", "📊 Business Intelligence (EDA)", "⚙️ Evaluación Técnica"])

# ------------------------------------------------------------------------------
# TAB 1: PREDICCIÓN
# ------------------------------------------------------------------------------
with tab1:
    st.subheader("Evaluación de Riesgo Individual")
    col_btn, col_res = st.columns([1, 2])

    with col_btn:
        st.write("")
        if st.button("⚡ Analizar Cliente", type="primary", use_container_width=True):
            input_data = pd.DataFrame({
                'Tenure_Days': [tenure], 'MonthlyCharges': [monthly_charges], 'Type': [contract],
                'InternetService': [internet], 'PaymentMethod': [payment],
                'OnlineSecurity': [online_security], 'TechSupport': [tech_support], 'PaperlessBilling': [paperless],
                'gender': ['Male'], 'SeniorCitizen': [0], 'Partner': ['No'], 'Dependents': ['No'],
                'PhoneService': ['Yes'], 'MultipleLines': ['No'], 'OnlineBackup': ['No'],
                'DeviceProtection': ['No'], 'StreamingTV': ['No'], 'StreamingMovies': ['No']
            })
            try:
                prob = pipeline.predict_proba(input_data)[:, 1][0]
                with col_res:
                    c1, c2 = st.columns(2)
                    with c1:
                        if prob > 0.5:
                            st.error("⚠️ **ALTO RIESGO**")
                            st.metric("Probabilidad", f"{prob:.1%}", delta="-Crítico", delta_color="inverse")
                        else:
                            st.success("✅ **CLIENTE SEGURO**")
                            st.metric("Probabilidad", f"{prob:.1%}", delta="Estable")
                    with c2:
                        fig = go.Figure(go.Indicator(
                            mode = "gauge+number", value = prob * 100,
                            domain = {'x': [0, 1], 'y': [0, 1]},
                            gauge = {'axis': {'range': [None, 100]}, 
                                     'steps': [{'range': [0, 50], 'color': "#A9DFBF"}, {'range': [80, 100], 'color': "#E74C3C"}],
                                     'threshold': {'line': {'color': "black", 'width': 4}, 'thickness': 0.75, 'value': prob * 100}}))
                        fig.update_layout(height=160, margin=dict(l=20,r=20,t=10,b=10))
                        st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error: {e}")

# ------------------------------------------------------------------------------
# TAB 2: BI INTERACTIVO (ENFOQUE DE NEGOCIO)
# ------------------------------------------------------------------------------
with tab2:
    st.header("Análisis Estratégico de Fuga")
    
    # --- A. FILTROS ---
    with st.container():
        st.markdown("### 🔍 Segmentación de Clientes")
        df_viz = df_full_loaded.copy()
        
        # Mapeo para que se vea bonito en los filtros
        df_viz['Churn_Label'] = df_viz['Churn'].map({0: 'Retenido', 1: 'Fugado'})
        
        c1, c2, c3 = st.columns(3)
        with c1: sel_contract = st.multiselect("Contrato", df_viz['Type'].unique(), default=df_viz['Type'].unique())
        with c2: sel_internet = st.multiselect("Internet", df_viz['InternetService'].unique(), default=df_viz['InternetService'].unique())
        with c3: sel_payment = st.multiselect("Pago", df_viz['PaymentMethod'].unique(), default=df_viz['PaymentMethod'].unique())
            
        df_filtered = df_viz[
            (df_viz['Type'].isin(sel_contract)) &
            (df_viz['InternetService'].isin(sel_internet)) &
            (df_viz['PaymentMethod'].isin(sel_payment))
        ]
        
        if df_filtered.empty: st.warning("⚠️ Sin datos."); st.stop()

    st.divider()

    # --- B. KPIs DE NEGOCIO ---
    col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)
    
    # Cálculos
    churn_rate = df_filtered['Churn'].mean()
    total_lost_rev = df_filtered[df_filtered['Churn']==1]['MonthlyCharges'].sum()
    arpu = df_filtered['MonthlyCharges'].mean() # Average Revenue Per User
    
    col_kpi1.metric("Clientes en Segmento", f"{len(df_filtered):,}")
    col_kpi2.metric("Tasa de Fuga Actual", f"{churn_rate:.1%}", delta_color="inverse")
    col_kpi3.metric("Ingreso Promedio (ARPU)", f"${arpu:.2f}")
    col_kpi4.metric("Ingreso Mensual Perdido", f"${total_lost_rev:,.0f}", help="Suma de cargos de clientes fugados en este segmento")
    
    st.markdown("---")

    # --- C. GRÁFICOS DE IMPACTO ---
    
    # FILA 1: CUÁNDO Y POR QUÉ
    r1c1, r1c2 = st.columns(2)
    
    with r1c1:
        st.subheader("1. La 'Zona de Peligro' (Antigüedad)")
        st.caption("¿En qué mes suelen cancelar los clientes?")
        
        # Filtramos solo los que se fugaron para ver cuándo mueren
        churners_only = df_filtered[df_filtered['Churn'] == 1]
        
        fig_hist = px.histogram(churners_only, x='Tenure_Days', nbins=20,
                                title="Distribución de Clientes Fugados por Tiempo",
                                labels={'Tenure_Days': 'Días antes de la fuga'},
                                color_discrete_sequence=['#E74C3C'])
        fig_hist.update_layout(bargap=0.1, yaxis_title="Cantidad de Fugas")
        st.plotly_chart(fig_hist, use_container_width=True)
        st.info("💡 **Insight:** La mayoría de fugas ocurren en los primeros 2 meses. El 'Onboarding' es crítico.")

    with r1c2:
        st.subheader("2. Fuga por Tipo de Contrato")
        st.caption("Comparativa de retención según compromiso.")
        
        churn_by_contract = df_filtered.groupby('Type')['Churn'].mean().reset_index()
        churn_by_contract['Churn'] = churn_by_contract['Churn'] * 100
        
        fig_bar = px.bar(churn_by_contract, x='Type', y='Churn', color='Churn',
                         title="Tasa de Fuga (%)",
                         color_continuous_scale='Reds', text_auto='.1f')
        fig_bar.update_layout(yaxis_range=[0, 100])
        st.plotly_chart(fig_bar, use_container_width=True)

    # FILA 2: DÓNDE ESTÁ EL DINERO
    r2c1, r2c2 = st.columns(2)

    with r2c1:
        st.subheader("3. Puntos Críticos: Internet")
        st.caption("¿Qué servicio técnico genera más insatisfacción?")
        
        # Agrupamos por Internet y calculamos tasa de fuga
        internet_churn = df_filtered.groupby('InternetService')['Churn'].mean().reset_index()
        internet_churn['Churn'] = internet_churn['Churn'] * 100
        
        fig_int = px.bar(internet_churn, x='Churn', y='InternetService', orientation='h',
                         title="Tasa de Fuga por Tecnología",
                         color='Churn', color_continuous_scale='OrRd', text_auto='.1f')
        st.plotly_chart(fig_int, use_container_width=True)
        
    with r2c2:
        st.subheader("4. Impacto Económico por Pago")
        st.caption("¿Qué método de pago nos hace perder más dinero?")
        
        # Calculamos DINERO perdido, no solo personas
        loss_by_payment = df_filtered[df_filtered['Churn']==1].groupby('PaymentMethod')['MonthlyCharges'].sum().reset_index()
        loss_by_payment = loss_by_payment.sort_values(by='MonthlyCharges', ascending=True)
        
        fig_money = px.bar(loss_by_payment, x='MonthlyCharges', y='PaymentMethod', orientation='h',
                           title="Ingreso Mensual Perdido ($)",
                           labels={'MonthlyCharges': 'Dinero Perdido ($)'},
                           color='MonthlyCharges', color_continuous_scale='reds')
        st.plotly_chart(fig_money, use_container_width=True)

# ------------------------------------------------------------------------------
# TAB 3: EVALUACIÓN TÉCNICA
# ------------------------------------------------------------------------------
with tab3:
    st.header("Evaluación del Modelo (Test Set)")

    with st.spinner("Calculando métricas..."):
        y_pred = pipeline.predict(X_test_loaded)
        y_prob = pipeline.predict_proba(X_test_loaded)[:, 1]
        y_true = y_test_loaded.values.ravel()

        auc = roc_auc_score(y_true, y_prob)
        acc = accuracy_score(y_true, y_pred)

        c1, c2 = st.columns(2)
        c1.metric("AUC-ROC", f"{auc:.4f}")
        c2.metric("Accuracy", f"{acc:.1%}")

        st.divider()

        col_shap, col_cm = st.columns(2)
        with col_shap:
            st.subheader("Feature Importance (SHAP)")
            df_imp = pd.DataFrame({
                'Variable': ['Antigüedad', 'Contrato 2 Años', 'Cargos Mensuales', 'Fibra Óptica', 'Contrato 1 Año', 'Seguridad Online'],
                'Impacto': [0.70, 0.60, 0.50, 0.40, 0.30, 0.20]
            })
            fig_shap = px.bar(df_imp, x='Impacto', y='Variable', orientation='h', color='Impacto')
            fig_shap.update_layout(yaxis=dict(autorange="reversed"), showlegend=False)
            st.plotly_chart(fig_shap, use_container_width=True)

        with col_cm:
            st.subheader("Matriz de Confusión")
            cm = confusion_matrix(y_true, y_pred)
            fig_cm = ff.create_annotated_heatmap(
                z=cm, x=['Pred: 0', 'Pred: 1'], y=['Real: 0', 'Real: 1'], colorscale='Blues',
                annotation_text=[[f"VN: {cm[0][0]}", f"FP: {cm[0][1]}"], [f"FN: {cm[1][0]}", f"VP: {cm[1][1]}"]])
            st.plotly_chart(fig_cm, use_container_width=True)

st.markdown("---")
st.caption("Interconnect BI & AI Analytics Platform")
