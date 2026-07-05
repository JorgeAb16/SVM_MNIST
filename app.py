import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    silhouette_score, adjusted_rand_score, accuracy_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
)

# ----------------------------------------------------------------------------
# Configuración general
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="PCA + K-Means + SVM | MNIST",
    page_icon="🔢",
    layout="wide",
    initial_sidebar_state="expanded",
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
    
    .pred-card.correcto {
        background: linear-gradient(135deg, #0b8a5e 0%, #15b87e 100%);
    }
    
    .pred-card.incorrecto {
        background: linear-gradient(135deg, #c0392b 0%, #e74c3c 100%);
    }
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a2332 0%, #1e2d42 50%, #243447 100%);
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }
    
    section[data-testid="stSidebar"] * {
        color: #e8ecf1 !important;
    }
    
    section[data-testid="stSidebar"] .stButton > button {
        background: linear-gradient(135deg, #2a5298 0%, #3a7bd5 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.7rem 1.5rem !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 6px 20px rgba(42, 82, 152, 0.4) !important;
    }
    
    section[data-testid="stSidebar"] .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 10px 28px rgba(42, 82, 152, 0.6) !important;
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
    
    /* Métricas con fondo oscuro */
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
    
    /* DataFrames */
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 2px 12px rgba(0,0,0,0.06);
    }
    
    /* Alertas */
    .stAlert {
        border-radius: 10px;
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
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="app-header">
        <h1>🔢 PCA + K-Means + SVM — Dígitos MNIST</h1>
        <p>Reducción de dimensionalidad, clustering y clasificación · IS-701 Inteligencia Artificial · Jorge Abraham Fajardo López · 20231900189</p>
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

# División train/test reproducible
df_train, df_test = train_test_split(
    df_muestra, test_size=0.25, random_state=RANDOM_STATE, stratify=df_muestra["digito_real"]
)
X_train_full = df_train[PCA_COLS].values
y_train = df_train["digito_real"].values
X_test_full = df_test[PCA_COLS].values
y_test = df_test["digito_real"].values

# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## ⚙️ Configuración")
    st.markdown("---")
    
    st.markdown("### 🎯 PCA")
    n_components = st.slider(
        "Componentes principales",
        min_value=2, max_value=N_MAX,
        value=min(30, N_MAX),
        help="Controla cuántas componentes de PCA se usan para K-Means y SVM.",
    )
    st.caption(f"📊 Disponibles: {N_MAX} componentes")
    
    st.markdown("---")
    st.markdown("### 🔍 Clasificación")
    idx_maximo = len(df_test) - 1
    idx_muestra = st.number_input(
        "Índice de muestra",
        min_value=0, max_value=idx_maximo,
        value=0, step=1,
        help="Selecciona una muestra del conjunto de prueba para clasificar."
    )
    clasificar = st.button("🔍 Clasificar Muestra", use_container_width=True, type="primary")

# ----------------------------------------------------------------------------
# Entrenamiento en vivo (cacheado por número de componentes)
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Entrenando K-Means y SVM...")
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


# ----------------------------------------------------------------------------
# Tabs
# ----------------------------------------------------------------------------
tab_pca, tab_kmeans, tab_svm, tab_comp = st.tabs(
    ["🧬 Análisis PCA", "📊 Clustering K-Means", "🧩 Clasificación SVM", "📈 Comparación"]
)

# --- Tab PCA ------------------------------------------------------------
with tab_pca:
    st.markdown("### 🧬 Varianza Explicada y Reconstrucción")
    st.markdown("---")
    
    col1, col2 = st.columns([1.3, 1])
    with col1:
        varianza_ac = np.cumsum(pca_full.explained_variance_ratio_)
        fig, ax = plt.subplots(figsize=(7, 4))
        fig.patch.set_facecolor('#f8f9fa')
        ax.set_facecolor('#f8f9fa')
        
        ax.plot(range(1, N_MAX + 1), varianza_ac, marker=".", color="#2a5298", linewidth=2, markersize=8)
        ax.axvline(n_components, color="#e74c3c", linestyle="--", linewidth=2, 
                   label=f"Seleccionado: {n_components}")
        ax.axhline(0.90, color="#95a5a6", linestyle=":", linewidth=2, label="90% varianza")
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
        fig2.patch.set_facecolor('#f8f9fa')
        axes[0].imshow(img_completa, cmap="gray")
        axes[0].set_title(f"Con {N_MAX} componentes", fontweight='bold')
        axes[0].axis("off")
        axes[1].imshow(img_reconstruida, cmap="gray")
        axes[1].set_title(f"Con {n_components} componentes", fontweight='bold')
        axes[1].axis("off")
        st.pyplot(fig2, use_container_width=True)
        
        st.info(
            f"💡 Reconstrucción del dígito real **{int(df_test.iloc[idx_muestra]['digito_real'])}** "
            f"con diferente número de componentes PCA. A menos componentes, más pérdida de detalle."
        )

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
    st.markdown("#### 🎨 Proyección 2D - Comparativa")
    
    col1, col2 = st.columns(2)
    with col1:
        fig3, ax3 = plt.subplots(figsize=(6, 5))
        fig3.patch.set_facecolor('#f8f9fa')
        ax3.set_facecolor('#f8f9fa')
        sns.scatterplot(data=df_viz, x="PC1", y="PC2", hue="digito_real", 
                       palette="tab10", s=25, alpha=.8, ax=ax3)
        ax3.set_title("Dígito Real", fontweight='bold', fontsize=14)
        ax3.legend(title="Dígito", bbox_to_anchor=(1.05, 1), loc='upper left')
        ax3.grid(True, alpha=0.3, linestyle='--')
        st.pyplot(fig3, use_container_width=True)
        
    with col2:
        fig4, ax4 = plt.subplots(figsize=(6, 5))
        fig4.patch.set_facecolor('#f8f9fa')
        ax4.set_facecolor('#f8f9fa')
        sns.scatterplot(data=df_viz, x="PC1", y="PC2", hue="cluster", 
                       palette="tab10", s=25, alpha=.8, ax=ax4)
        ax4.set_title("Cluster K-Means", fontweight='bold', fontsize=14)
        ax4.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc='upper left')
        ax4.grid(True, alpha=0.3, linestyle='--')
        st.pyplot(fig4, use_container_width=True)

    st.markdown("---")
    st.markdown("#### 🔢 Matriz de Confusión: Dígito Real vs Cluster")
    tabla_cruzada = pd.crosstab(y_train, resultados["clusters_tr"],
                                 rownames=["Dígito real"], colnames=["Cluster"])
    fig5, ax5 = plt.subplots(figsize=(10, 6))
    fig5.patch.set_facecolor('#f8f9fa')
    ax5.set_facecolor('#f8f9fa')
    sns.heatmap(tabla_cruzada, annot=True, fmt="d", cmap="YlOrRd", 
                ax=ax5, cbar_kws={'label': 'Cantidad'}, linewidths=0.5)
    ax5.set_title("Distribución de dígitos por cluster", fontweight='bold', fontsize=14)
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

    if clasificar or True:
        fila_pca = df_test.iloc[idx_muestra][PCA_COLS].values[:n_components].reshape(1, -1)
        digito_real = int(df_test.iloc[idx_muestra]["digito_real"])
        digito_predicho = int(resultados["svm"].predict(fila_pca)[0])
        acierto = digito_real == digito_predicho

        st.markdown("---")
        st.markdown("#### 🔍 Clasificación de Muestra")
        
        col1, col2 = st.columns([1, 2])
        with col1:
            img = reconstruir_imagen(df_test.iloc[idx_muestra][PCA_COLS].values, n_components)
            fig6, ax6 = plt.subplots(figsize=(4, 4))
            fig6.patch.set_facecolor('#f8f9fa')
            ax6.imshow(img, cmap="gray")
            ax6.set_title(f"Dígito real: {digito_real}", fontweight='bold')
            ax6.axis("off")
            st.pyplot(fig6, use_container_width=True)
            
        with col2:
            card_class = "correcto" if acierto else "incorrecto"
            icono = "✅" if acierto else "❌"
            texto = "Correcto" if acierto else "Incorrecto"
            
            st.markdown(
                f"""
                <div class="pred-card {card_class}">
                    <div style="font-size:1.1rem; margin-bottom:0.5rem;">Dígito predicho por SVM</div>
                    <div class="digit">{digito_predicho}</div>
                    <div style="margin-top:1rem; font-size:1rem;">
                        {icono} {texto} — Dígito real: {digito_real}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("---")
    st.markdown("#### 📋 Reporte de Clasificación")
    df_reporte = pd.DataFrame(resultados["reporte"]).transpose().round(3)
    st.dataframe(df_reporte, use_container_width=True)

    st.markdown("---")
    st.markdown("#### 🎯 Matriz de Confusión")
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
        st.markdown("#### 📊 Datos de comparación")
        st.dataframe(df_comparacion, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown("#### 📉 Métricas vs Componentes")
        
        fig8, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig8.patch.set_facecolor('#f8f9fa')
        
        axes[0].set_facecolor('#f8f9fa')
        sns.lineplot(data=df_comparacion, x="n_componentes", y="accuracy", 
                    marker="o", ax=axes[0], color="#2a5298", linewidth=2, markersize=8)
        axes[0].axvline(n_components, color="#e74c3c", linestyle="--", linewidth=2, 
                       label=f"Actual: {n_components}")
        axes[0].set_title("Accuracy vs Componentes", fontweight='bold', fontsize=12)
        axes[0].set_xlabel("Número de componentes", fontweight='bold')
        axes[0].set_ylabel("Accuracy", fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3, linestyle='--')
        
        axes[1].set_facecolor('#f8f9fa')
        sns.lineplot(data=df_comparacion, x="n_componentes", y="tiempo_entrenamiento_seg",
                    marker="o", color="#e67e22", ax=axes[1], linewidth=2, markersize=8)
        axes[1].set_title("Tiempo de Entrenamiento vs Componentes", fontweight='bold', fontsize=12)
        axes[1].set_xlabel("Número de componentes", fontweight='bold')
        axes[1].set_ylabel("Tiempo (segundos)", fontweight='bold')
        axes[1].grid(True, alpha=0.3, linestyle='--')
        
        st.pyplot(fig8, use_container_width=True)

    if metadata is not None:
        st.markdown("---")
        with st.expander("🔬 Metadatos del Modelo"):
            st.json(metadata)

# Footer
st.markdown("---")
st.markdown(
    """
    <div class="footer">
        <strong>IS-701 Inteligencia Artificial</strong> · UNAH Campus Comayagua<br>
        <span style="opacity:0.7;">Proyecto Académico · PCA + K-Means + SVM · MNIST Digits</span>
    </div>
    """,
    unsafe_allow_html=True,
)
