"""
App de Streamlit - Reducción de Dimensionalidad y Clasificación con PCA, K-Means y SVM
Dataset: MNIST Digit Recognizer (Kaggle)
Asignatura: IS-701 - Inteligencia Artificial - Campus Comayagua

Estructura de carpetas esperada (generada por el notebook
Actividad_5_Casa_PCA_KMeans_SVM_MNIST.ipynb):

    app.py
    requirements.txt
    models/
        scaler_mnist.pkl
        pca_mnist.pkl
        kmeans_mnist.pkl
        svm_mnist.pkl
        model_metadata.json
    outputs/
        mnist_comparacion_componentes.csv
        mnist_muestra_clusters.csv
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    silhouette_score, adjusted_rand_score, accuracy_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
)

try:
    from streamlit_drawable_canvas import st_canvas
    CANVAS_DISPONIBLE = True
except ImportError:
    CANVAS_DISPONIBLE = False

# ----------------------------------------------------------------------------
# Configuración general
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Dibuja y Clasifica | MNIST",
    page_icon="✏️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

MODEL_DIR = Path("models")
OUTPUT_DIR = Path("outputs")

SCALER_PATH = MODEL_DIR / "scaler_mnist.pkl"
PCA_PATH = MODEL_DIR / "pca_mnist.pkl"
KMEANS_PATH = MODEL_DIR / "kmeans_mnist.pkl"
SVM_PATH = MODEL_DIR / "svm_mnist.pkl"
META_PATH = MODEL_DIR / "model_metadata.json"
COMPARACION_PATH = OUTPUT_DIR / "mnist_comparacion_componentes.csv"
CLUSTERS_PATH = OUTPUT_DIR / "mnist_muestra_clusters.csv"

RANDOM_STATE = 42

# ----------------------------------------------------------------------------
# Estilos personalizados
# ----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #e8ecf1 100%);
        background-attachment: fixed;
    }
    
    /* Header principal */
    .app-header {
        padding: 2rem 2.5rem;
        border-radius: 20px;
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 50%, #3a7bd5 100%);
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 15px 40px rgba(30, 60, 114, 0.3);
        position: relative;
        overflow: hidden;
        text-align: center;
    }
    
    .app-header::before {
        content: '';
        position: absolute;
        top: -30%;
        right: -20%;
        width: 150%;
        height: 150%;
        background: radial-gradient(circle, rgba(255,255,255,0.08) 0%, transparent 70%);
    }
    
    .app-header h1 {
        margin-bottom: 0.3rem;
        font-size: 2rem;
        font-weight: 700;
        position: relative;
        z-index: 1;
    }
    
    .app-header p {
        margin: 0;
        opacity: 0.9;
        font-size: 1rem;
        font-weight: 300;
        position: relative;
        z-index: 1;
    }
    
    /* Tarjeta de predicción */
    .pred-card {
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        color: white;
        background: linear-gradient(135deg, #2c3e50 0%, #3498db 100%);
        box-shadow: 0 15px 35px rgba(44, 62, 80, 0.3);
        transition: transform 0.3s ease;
    }
    
    .pred-card:hover {
        transform: translateY(-3px);
    }
    
    .pred-card .digit { 
        font-size: 4rem; 
        font-weight: 700;
        text-shadow: 0 4px 12px rgba(0,0,0,0.3);
    }
    
    /* Pestañas */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #1e3c72 !important;
        padding: 8px 12px;
        border-radius: 16px;
        box-shadow: 0 4px 15px rgba(30, 60, 114, 0.2);
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        padding: 12px 24px;
        font-weight: 500;
        color: rgba(255, 255, 255, 0.8) !important;
        background-color: transparent;
        border: none;
        transition: all 0.3s ease;
        font-size: 0.95rem;
    }
    
    .stTabs [aria-selected="true"] {
        background: rgba(255, 255, 255, 0.2) !important;
        color: white !important;
        font-weight: 600;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background-color: rgba(255, 255, 255, 0.1) !important;
        color: white !important;
    }
    
    /* Métricas */
    [data-testid="stMetric"] {
        background-color: #2c3e50 !important;
        padding: 1.2rem !important;
        border-radius: 12px !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15) !important;
        border: 1px solid #34495e !important;
    }
    
    [data-testid="stMetric"] label {
        color: #bdc3c7 !important;
        font-weight: 500 !important;
        font-size: 0.85rem !important;
    }
    
    [data-testid="stMetricValue"] {
        color: #ecf0f1 !important;
        font-weight: 700 !important;
        font-size: 1.8rem !important;
    }
    
    /* Canvas */
    .canvas-container {
        background: white;
        border-radius: 16px;
        padding: 1rem;
        box-shadow: 0 8px 30px rgba(0,0,0,0.1);
        border: 2px solid #e1e5eb;
    }
    
    /* DataFrames */
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 2px 12px rgba(0,0,0,0.06);
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 1.5rem;
        background: linear-gradient(135deg, #1a2332 0%, #1e2d42 100%);
        border-radius: 12px;
        margin-top: 2rem;
        color: #a0aec0;
        font-size: 0.9rem;
    }
    
    /* Ocultar sidebar */
    section[data-testid="stSidebar"] {
        display: none;
    }
    
    /* Ajustar contenido principal */
    .main .block-container {
        max-width: 100%;
        padding-left: 1rem;
        padding-right: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="app-header">
        <h1>✏️ Dibuja un Dígito y Clasifícalo</h1>
        <p>PCA + K-Means + SVM · MNIST · IS-701 Inteligencia Artificial · UNAH Campus Comayagua</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# Carga de modelos y datos
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def cargar_pkl(path):
    return joblib.load(path) if path.exists() else None


@st.cache_data(show_spinner=False)
def cargar_metadata():
    if not META_PATH.exists():
        return None
    with open(META_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def cargar_csv(path):
    return pd.read_csv(path) if path.exists() else None


scaler = cargar_pkl(SCALER_PATH)
pca_full = cargar_pkl(PCA_PATH)
kmeans_guardado = cargar_pkl(KMEANS_PATH)
svm_guardado = cargar_pkl(SVM_PATH)
metadata = cargar_metadata()
df_comparacion = cargar_csv(COMPARACION_PATH)
df_muestra = cargar_csv(CLUSTERS_PATH)

archivos_faltantes = [str(p) for p in [SCALER_PATH, PCA_PATH, META_PATH, CLUSTERS_PATH] if not p.exists()]
if archivos_faltantes:
    st.warning(
        "⚠️ No se encontraron algunos archivos necesarios: " + ", ".join(archivos_faltantes) +
        ". Copia las carpetas `models/` y `outputs/` generadas por el notebook junto a `app.py`."
    )
    st.stop()

N_MAX = pca_full.n_components_
PCA_COLS = [f"PC{i+1}" for i in range(N_MAX)]

# Parámetros fijos (sin sidebar)
n_components = min(30, N_MAX)
idx_muestra = 0

# División train/test reproducible
df_train, df_test = train_test_split(
    df_muestra, test_size=0.25, random_state=RANDOM_STATE, stratify=df_muestra["digito_real"]
)
X_train_full = df_train[PCA_COLS].values
y_train = df_train["digito_real"].values
X_test_full = df_test[PCA_COLS].values
y_test = df_test["digito_real"].values

# ----------------------------------------------------------------------------
# Entrenamiento en vivo
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Cargando modelos...")
def entrenar_modelos(n_comp):
    X_tr = X_train_full[:, :n_comp]
    X_te = X_test_full[:, :n_comp]

    km = KMeans(n_clusters=10, random_state=RANDOM_STATE, n_init=10)
    clusters_tr = km.fit_predict(X_tr)
    sil = silhouette_score(X_tr, clusters_tr)
    ari = adjusted_rand_score(y_train, clusters_tr)

    svm = SVC(kernel="rbf", C=10, gamma="scale", random_state=RANDOM_STATE)
    svm.fit(X_tr, y_train)
    y_pred = svm.predict(X_te)
    acc = accuracy_score(y_test, y_pred)
    reporte = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    varianza_acumulada = float(np.cumsum(pca_full.explained_variance_ratio_)[n_comp - 1])

    return {
        "kmeans": km, "clusters_tr": clusters_tr, "silhouette": sil, "ari": ari,
        "svm": svm, "y_pred": y_pred, "accuracy": acc, "reporte": reporte,
        "varianza_acumulada": varianza_acumulada,
    }


resultados = entrenar_modelos(n_components)


def reconstruir_imagen(fila_pca, n_comp):
    """Reconstruye una imagen 28x28 a partir de sus componentes PCA."""
    vector = np.zeros(N_MAX)
    vector[:n_comp] = fila_pca[:n_comp]
    x_escalado = pca_full.inverse_transform(vector.reshape(1, -1))
    x_pixeles = scaler.inverse_transform(x_escalado)
    return x_pixeles.reshape(28, 28)


def procesar_trazo_dibujado(image_data):
    """Convierte el arreglo RGBA del canvas en un vector de 784 píxeles."""
    img = Image.fromarray(image_data.astype("uint8"), mode="RGBA").convert("L")
    img = img.resize((28, 28), Image.LANCZOS)
    arr = np.array(img).astype("float32")
    arr_invertido = 255.0 - arr
    return arr_invertido, arr_invertido.flatten().reshape(1, -1)


# ----------------------------------------------------------------------------
# Tabs - Dibujo como principal
# ----------------------------------------------------------------------------
tab_dibujo, tab_pca, tab_kmeans, tab_svm, tab_comp = st.tabs(
    ["✏️ Dibujar Dígito", "🧬 PCA", "📊 K-Means", "🧩 Clasificación SVM", "📈 Comparación"]
)

# --- Tab Dibujar dígito (PRINCIPAL) -------------------------------------------
with tab_dibujo:
    st.markdown(f"### ✏️ Dibuja un dígito y el modelo lo clasificará")
    st.markdown("---")

    if not CANVAS_DISPONIBLE:
        st.error(
            "❌ Falta instalar `streamlit-drawable-canvas`. Ejecuta:\n"
            "```bash\npip install streamlit-drawable-canvas\n```"
        )
    else:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("#### 🎨 Lienzo de dibujo")
            st.caption("Dibuja un dígito (0-9) con el mouse o dedo.")
            
            canvas_result = st_canvas(
                fill_color="rgba(0, 0, 0, 1)",
                stroke_width=20,
                stroke_color="#000000",
                background_color="#FFFFFF",
                height=300,
                width=300,
                drawing_mode="freedraw",
                key="canvas_digito",
            )
            
            col_btn1, col_btn2 = st.columns(2)
            clasificar_dibujo = col_btn1.button(
                "🔍 Clasificar Dibujo", 
                use_container_width=True, 
                type="primary"
            )
            col_btn2.caption("🗑️ Usa el ícono de borrar en la esquina del lienzo para limpiar.")

        with col2:
            st.markdown("#### 🤖 Resultado de la clasificación")
            
            hay_trazo = (
                canvas_result.image_data is not None
                and canvas_result.image_data[:, :, :3].min() < 255
            )
            
            if clasificar_dibujo and hay_trazo:
                with st.spinner("Procesando dibujo..."):
                    imagen_28, vector_pixeles = procesar_trazo_dibujado(canvas_result.image_data)
                    vector_escalado = scaler.transform(vector_pixeles)
                    vector_pca_full = pca_full.transform(vector_escalado)
                    vector_pca = vector_pca_full[:, :n_components]

                    digito_predicho = int(resultados["svm"].predict(vector_pca)[0])
                    cluster_asignado = int(resultados["kmeans"].predict(vector_pca)[0])

                st.markdown("**📷 Imagen procesada (28×28):**")
                fig9, ax9 = plt.subplots(figsize=(2.6, 2.6))
                ax9.imshow(imagen_28, cmap="gray")
                ax9.axis("off")
                st.pyplot(fig9, use_container_width=False)

                st.markdown(
                    f"""
                    <div class="pred-card">
                        <div style="font-size:1.1rem;">Dígito predicho por SVM</div>
                        <div class="digit">{digito_predicho}</div>
                        <div style="margin-top:0.8rem; font-size:0.9rem;">
                            📊 Cluster K-Means: {cluster_asignado}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                
            elif clasificar_dibujo and not hay_trazo:
                st.warning("⚠️ El lienzo está vacío. Dibuja un dígito antes de clasificar.")
            else:
                st.info("💡 Dibuja un dígito en el lienzo y presiona **Clasificar Dibujo** para ver la predicción del modelo.")

        st.markdown("---")
        st.info(
            "💡 **Nota:** El modelo fue entrenado con dígitos del dataset MNIST (centrados, trazo uniforme). "
            "Un dibujo a mano libre puede verse diferente, por lo que la predicción podría variar. "
            "¡Es un excelente ejemplo de cómo el modelo generaliza fuera de su distribución de entrenamiento!"
        )

# --- Tab PCA ------------------------------------------------------------
with tab_pca:
    st.markdown(f"### 🧬 Varianza Explicada y Reconstrucción ({n_components} componentes)")
    st.markdown("---")

    col1, col2 = st.columns([1.3, 1])
    with col1:
        varianza_ac = np.cumsum(pca_full.explained_variance_ratio_)
        fig, ax = plt.subplots(figsize=(7, 4))
        fig.patch.set_facecolor('#f8f9fa')
        ax.set_facecolor('#f8f9fa')
        ax.plot(range(1, N_MAX + 1), varianza_ac, marker=".", color="#2a5298", linewidth=2)
        ax.axvline(n_components, color="red", linestyle="--", linewidth=2, label=f"Usando: {n_components}")
        ax.axhline(0.90, color="gray", linestyle=":", linewidth=2, label="90% varianza")
        ax.set_xlabel("Número de componentes", fontweight='bold')
        ax.set_ylabel("Varianza explicada acumulada", fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, linestyle='--')
        st.pyplot(fig, use_container_width=True)
        st.metric("📊 Varianza explicada", f"{resultados['varianza_acumulada']*100:.2f}%")

    with col2:
        fila = df_test.iloc[idx_muestra][PCA_COLS].values
        img_reconstruida = reconstruir_imagen(fila, n_components)
        img_completa = reconstruir_imagen(fila, N_MAX)

        fig2, axes = plt.subplots(1, 2, figsize=(6, 3.2))
        axes[0].imshow(img_completa, cmap="gray")
        axes[0].set_title(f"Con {N_MAX} comp.", fontweight='bold')
        axes[0].axis("off")
        axes[1].imshow(img_reconstruida, cmap="gray")
        axes[1].set_title(f"Con {n_components} comp.", fontweight='bold')
        axes[1].axis("off")
        st.pyplot(fig2, use_container_width=True)
        st.info(f"💡 Reconstrucción del dígito real **{int(df_test.iloc[idx_muestra]['digito_real'])}** con diferente número de componentes.")

# --- Tab K-Means ----------------------------------------------------------
with tab_kmeans:
    st.markdown(f"### 📊 Agrupamiento K-Means (K=10, {n_components} componentes)")
    st.markdown("---")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("🎯 Silhouette Score", f"{resultados['silhouette']:.4f}")
    with c2:
        st.metric("📐 Adjusted Rand Index", f"{resultados['ari']:.4f}")
    with c3:
        st.metric("📊 Muestras entrenamiento", f"{len(X_train_full):,}")

    df_viz = pd.DataFrame(X_train_full[:, :2], columns=["PC1", "PC2"])
    df_viz["digito_real"] = y_train.astype(str)
    df_viz["cluster"] = resultados["clusters_tr"].astype(str)

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        fig3, ax3 = plt.subplots(figsize=(6, 5))
        fig3.patch.set_facecolor('#f8f9fa')
        ax3.set_facecolor('#f8f9fa')
        sns.scatterplot(data=df_viz, x="PC1", y="PC2", hue="digito_real", palette="tab10", s=20, alpha=.7, ax=ax3)
        ax3.set_title("Proyección 2D — dígito real", fontweight='bold')
        ax3.legend(title="Dígito", bbox_to_anchor=(1.05, 1), loc='upper left')
        st.pyplot(fig3, use_container_width=True)
    with col2:
        fig4, ax4 = plt.subplots(figsize=(6, 5))
        fig4.patch.set_facecolor('#f8f9fa')
        ax4.set_facecolor('#f8f9fa')
        sns.scatterplot(data=df_viz, x="PC1", y="PC2", hue="cluster", palette="tab10", s=20, alpha=.7, ax=ax4)
        ax4.set_title("Proyección 2D — cluster (K-Means)", fontweight='bold')
        ax4.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc='upper left')
        st.pyplot(fig4, use_container_width=True)

    st.markdown("---")
    st.markdown("#### 🔢 Dígito real vs. cluster asignado")
    tabla_cruzada = pd.crosstab(y_train, resultados["clusters_tr"],
                                 rownames=["Dígito real"], colnames=["Cluster"])
    fig5, ax5 = plt.subplots(figsize=(10, 6))
    fig5.patch.set_facecolor('#f8f9fa')
    ax5.set_facecolor('#f8f9fa')
    sns.heatmap(tabla_cruzada, annot=True, fmt="d", cmap="YlOrRd", ax=ax5, linewidths=0.5)
    st.pyplot(fig5, use_container_width=True)

# --- Tab SVM ----------------------------------------------------------------
with tab_svm:
    st.markdown(f"### 🧩 Clasificación SVM (RBF Kernel, {n_components} componentes)")
    st.markdown("---")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("🎯 Accuracy (test)", f"{resultados['accuracy']*100:.2f}%")
    with c2:
        st.metric("📊 F1-Score Macro", f"{resultados['reporte']['macro avg']['f1-score']*100:.2f}%")
    with c3:
        st.metric("🔢 Muestras test", f"{len(X_test_full):,}")

    fila_pca = df_test.iloc[idx_muestra][PCA_COLS].values[:n_components].reshape(1, -1)
    digito_real = int(df_test.iloc[idx_muestra]["digito_real"])
    digito_predicho = int(resultados["svm"].predict(fila_pca)[0])

    st.markdown("---")
    col1, col2 = st.columns([1, 2])
    with col1:
        img = reconstruir_imagen(df_test.iloc[idx_muestra][PCA_COLS].values, n_components)
        fig6, ax6 = plt.subplots(figsize=(3, 3))
        ax6.imshow(img, cmap="gray")
        ax6.set_title(f"Dígito real: {digito_real}", fontweight='bold')
        ax6.axis("off")
        st.pyplot(fig6, use_container_width=True)
    with col2:
        acierto = "✅ Correcto" if digito_real == digito_predicho else "❌ Incorrecto"
        st.markdown(
            f"""
            <div class="pred-card">
                <div>Dígito predicho por el SVM</div>
                <div class="digit">{digito_predicho}</div>
                <div style="margin-top:0.5rem; font-size:0.95rem;">
                    Dígito real: {digito_real} · {acierto}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("#### 📋 Reporte de clasificación")
    df_reporte = pd.DataFrame(resultados["reporte"]).transpose().round(3)
    st.dataframe(df_reporte, use_container_width=True)

    st.markdown("#### 🎯 Matriz de confusión")
    fig7, ax7 = plt.subplots(figsize=(8, 8))
    fig7.patch.set_facecolor('#f8f9fa')
    ax7.set_facecolor('#f8f9fa')
    ConfusionMatrixDisplay.from_predictions(
        y_test, resultados["y_pred"], cmap="Blues", ax=ax7, colorbar=True
    )
    ax7.set_title("Matriz de Confusión - SVM", fontweight='bold', fontsize=14)
    st.pyplot(fig7, use_container_width=True)

# --- Tab Comparación ----------------------------------------------------------
with tab_comp:
    st.markdown("### 📈 Efecto de la Reducción de Dimensionalidad")
    st.markdown("---")

    if df_comparacion is None:
        st.warning("📁 No se encontró `outputs/mnist_comparacion_componentes.csv`.")
    else:
        st.dataframe(df_comparacion, use_container_width=True, hide_index=True)

        fig8, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig8.patch.set_facecolor('#f8f9fa')
        
        axes[0].set_facecolor('#f8f9fa')
        sns.lineplot(data=df_comparacion, x="n_componentes", y="accuracy", marker="o", ax=axes[0], color="#2a5298", linewidth=2)
        axes[0].axvline(n_components, color="red", linestyle="--", label=f"Actual: {n_components}")
        axes[0].set_title("Accuracy vs. número de componentes", fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3, linestyle='--')
        
        axes[1].set_facecolor('#f8f9fa')
        sns.lineplot(data=df_comparacion, x="n_componentes", y="tiempo_entrenamiento_seg",
                     marker="o", color="#e67e22", ax=axes[1], linewidth=2)
        axes[1].set_title("Tiempo de entrenamiento vs. número de componentes", fontweight='bold')
        axes[1].grid(True, alpha=0.3, linestyle='--')
        
        st.pyplot(fig8, use_container_width=True)

    if metadata is not None:
        with st.expander("🔬 Metadatos del modelo"):
            st.json(metadata)

# Footer
st.markdown("---")
st.markdown(
    """
    <div class="footer">
        <strong>IS-701 Inteligencia Artificial</strong> · UNAH Campus Comayagua<br>
        <span style="opacity:0.7;">Jorge Abraham Fajardo López · 20231900189</span>
    </div>
    """,
    unsafe_allow_html=True,
)
