# 📡 Interconnect Telecom: Churn Prediction & Retention Dashboard

[![Streamlit App]([https://static.streamlit.io/badges/streamlit_badge_black_white.svg](https://telecom-ta8rdbryzmacvuwadjzwky.streamlit.app/))]
![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Library](https://img.shields.io/badge/Library-Scikit--Learn%20|%20LightGBM-orange)
![Deployment](https://img.shields.io/badge/Deployment-Streamlit-red)

## 📄 Descripción del Proyecto

Este proyecto consiste en el desarrollo de una solución integral de Machine Learning para predecir la fuga de clientes (Churn) en una empresa de telecomunicaciones. El objetivo principal es identificar usuarios en riesgo de cancelación para implementar estrategias de retención proactivas.

El sistema no se limita al modelado; incluye un **Dashboard Interactivo desplegado en la nube** que permite a los equipos de negocio simular escenarios, visualizar patrones históricos y evaluar el rendimiento del modelo en tiempo real.

### 🚀 Demo en Vivo
¡Prueba la aplicación interactiva aquí!
👉 **[https://telecom-ta8rdbryzmacvuwadjzwky.streamlit.app/]**

---

## 🛠️ Características Principales

El proyecto abarca un pipeline *End-to-End* que incluye:

1.  **ETL & Preprocesamiento:** Limpieza de datos, ingeniería de características (Feature Engineering) y manejo de desbalance de clases.
2.  **Modelado Predictivo:** Entrenamiento y optimización de algoritmos de Gradient Boosting (LightGBM, XGBoost) y Random Forest.
3.  **Dashboard de Business Intelligence (BI):**
    * **Simulador de Riesgo:** Predicción de probabilidad de fuga en tiempo real ajustando parámetros del cliente.
    * **EDA Interactivo:** Visualización de patrones de negocio (contratos, métodos de pago, antigüedad) con filtros dinámicos.
    * **Evaluación Técnica:** Visualización de métricas de rendimiento (AUC-ROC, Matriz de Confusión) e importancia de variables (SHAP).

---

## 📊 Rendimiento del Modelo

El modelo final seleccionado fue **LightGBM** debido a su eficiencia y alto rendimiento en métricas de clasificación.

| Métrica | Resultado (Test Set) | Descripción |
| :--- | :--- | :--- |
| **AUC-ROC** | **0.9054** | Excelente capacidad de distinción entre clases. |
| **Accuracy** | ~85% | Precisión global del modelo. |

**Insights Clave (Factores Determinantes):**
* **Antigüedad (Tenure):** Clientes nuevos (< 2 meses) tienen el mayor riesgo.
* **Tipo de Contrato:** Los contratos mensuales ("Month-to-month") son el predictor más fuerte de fuga.
* **Método de Pago:** El uso de "Electronic Check" está altamente correlacionado con la cancelación.
* **Servicio de Internet:** Usuarios de Fibra Óptica presentan mayor tasa de churn.

---

## 💻 Stack Tecnológico

* **Lenguaje:** Python
* **Manipulación de Datos:** Pandas, NumPy
* **Machine Learning:** Scikit-learn, LightGBM, Joblib
* **Visualización:** Plotly Express, Plotly Graph Objects (Interactivo)
* **Despliegue Web:** Streamlit
* **Control de Versiones:** Git / GitHub

---

## 📂 Estructura del Proyecto

```text
Telecom/
├── app_dashboard/          # CARPETA DE DESPLIEGUE (Streamlit)
│   ├── app.py              # Código fuente del Dashboard
│   ├── requirements.txt    # Dependencias para la nube
│   ├── modelo_churn...pkl  # Modelo entrenado (Joblib)
│   └── *.csv               # Datos de prueba y análisis (Test sets)
│
├── Notebooks/              # JUPYTER NOTEBOOKS
│   └── Churn_Analysis.ipynb # EDA, Preprocesamiento y Entrenamiento
│
└── Data/                   # DATASETS ORIGINALES
    └── ...
