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
# Estilos
# ----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .app-header {
        padding: 1.6rem 2rem;
        border-radius: 16px;
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 45%, #0f3460 100%);
        color: white;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 24px rgba(15, 20, 40, 0.3);
    }
    .app-header h1 { margin-bottom: 0.2rem; font-size: 1.9rem; }
    .app-header p { margin: 0; opacity: 0.9; font-size: 0.95rem; }
    .metric-card {
        background: white;
        border-radius: 14px;
        padding: 1rem 1.2rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.06);
        border: 1px solid #eaeef3;
        text-align: center;
    }
    .pred-card {
        padding: 1.8rem;
        border-radius: 16px;
        text-align: center;
        color: white;
        background: linear-gradient(135deg, #0f3460, #533483);
        box-shadow: 0 6px 18px rgba(0,0,0,0.2);
    }
    .pred-card .digit { font-size: 3.2rem; font-weight: 700; }
    section[data-testid="stSidebar"] { background-color: #16213e; }
    section[data-testid="stSidebar"] * { color: #f0f3f8 !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="app-header">
        <h1>🔢 PCA + K-Means + SVM — Dígitos MNIST</h1>
        <p>IS-701 · Inteligencia Artificial · Campus Comayagua · Reducción de dimensionalidad, clustering y clasificación</p>
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

# División train/test reproducible dentro de la app, para poder calcular métricas
# en vivo según el número de componentes que el usuario seleccione.
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
    st.header("⚙️ Parámetros")
    n_components = st.slider(
        "Número de componentes principales (PCA)",
        min_value=2, max_value=N_MAX,
        value=min(30, N_MAX),
        help="Controla cuántas componentes de PCA se usan para el clustering (K-Means) y la clasificación (SVM).",
    )
    st.caption(f"Componentes disponibles guardadas: {N_MAX}")

    st.markdown("---")
    idx_maximo = len(df_test) - 1
    idx_muestra = st.number_input(
        "Índice de muestra de prueba a clasificar", min_value=0, max_value=idx_maximo, value=0, step=1
    )
    clasificar = st.button("🔍 Clasificar muestra", use_container_width=True, type="primary")

# ----------------------------------------------------------------------------
# Entrenamiento en vivo (cacheado por número de componentes)
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Entrenando K-Means y SVM con el número de componentes seleccionado...")
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
    """Reconstruye una imagen 28x28 a partir de sus componentes PCA, usando solo las
    primeras n_comp (el resto se pone en cero, como si no se hubieran usado)."""
    vector = np.zeros(N_MAX)
    vector[:n_comp] = fila_pca[:n_comp]
    x_escalado = pca_full.inverse_transform(vector.reshape(1, -1))
    x_pixeles = scaler.inverse_transform(x_escalado)
    return x_pixeles.reshape(28, 28)


# ----------------------------------------------------------------------------
# Tabs
# ----------------------------------------------------------------------------
tab_pca, tab_kmeans, tab_svm, tab_comp = st.tabs(
    ["🧬 PCA", "📊 K-Means", "🧩 Clasificación (SVM)", "📈 Comparación"]
)

# --- Tab PCA ------------------------------------------------------------
with tab_pca:
    st.subheader("Varianza explicada y reconstrucción de imagen")

    col1, col2 = st.columns([1.3, 1])
    with col1:
        varianza_ac = np.cumsum(pca_full.explained_variance_ratio_)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(range(1, N_MAX + 1), varianza_ac, marker=".")
        ax.axvline(n_components, color="red", linestyle="--", label=f"Seleccionado: {n_components}")
        ax.axhline(0.90, color="gray", linestyle=":", label="90% varianza")
        ax.set_xlabel("Número de componentes")
        ax.set_ylabel("Varianza explicada acumulada")
        ax.legend()
        st.pyplot(fig, use_container_width=True)
        st.metric("Varianza explicada con los componentes seleccionados",
                   f"{resultados['varianza_acumulada']*100:.2f}%")

    with col2:
        fila = df_test.iloc[idx_muestra][PCA_COLS].values
        img_reconstruida = reconstruir_imagen(fila, n_components)
        img_completa = reconstruir_imagen(fila, N_MAX)

        fig2, axes = plt.subplots(1, 2, figsize=(6, 3.2))
        axes[0].imshow(img_completa, cmap="gray")
        axes[0].set_title(f"Con {N_MAX} comp.")
        axes[0].axis("off")
        axes[1].imshow(img_reconstruida, cmap="gray")
        axes[1].set_title(f"Con {n_components} comp.")
        axes[1].axis("off")
        st.pyplot(fig2, use_container_width=True)
        st.caption(
            f"Reconstrucción del dígito real **{int(df_test.iloc[idx_muestra]['digito_real'])}** "
            "(muestra seleccionada en la barra lateral) a partir de sus componentes PCA. "
            "A menos componentes, más se pierde el detalle de la imagen original."
        )

# --- Tab K-Means ----------------------------------------------------------
with tab_kmeans:
    st.subheader(f"Agrupamiento con K-Means (K=10) usando {n_components} componentes")

    c1, c2, c3 = st.columns(3)
    c1.metric("Silhouette score", round(resultados["silhouette"], 4))
    c2.metric("Adjusted Rand Index", round(resultados["ari"], 4))
    c3.metric("Muestras de entrenamiento", len(X_train_full))

    df_viz = pd.DataFrame(X_train_full[:, :2], columns=["PC1", "PC2"])
    df_viz["digito_real"] = y_train.astype(str)
    df_viz["cluster"] = resultados["clusters_tr"].astype(str)

    col1, col2 = st.columns(2)
    with col1:
        fig3, ax3 = plt.subplots(figsize=(6, 5))
        sns.scatterplot(data=df_viz, x="PC1", y="PC2", hue="digito_real", palette="tab10", s=20, alpha=.7, ax=ax3)
        ax3.set_title("Proyección 2D — dígito real")
        st.pyplot(fig3, use_container_width=True)
    with col2:
        fig4, ax4 = plt.subplots(figsize=(6, 5))
        sns.scatterplot(data=df_viz, x="PC1", y="PC2", hue="cluster", palette="tab10", s=20, alpha=.7, ax=ax4)
        ax4.set_title("Proyección 2D — cluster (K-Means)")
        st.pyplot(fig4, use_container_width=True)

    st.markdown("#### Dígito real vs. cluster asignado")
    tabla_cruzada = pd.crosstab(y_train, resultados["clusters_tr"],
                                 rownames=["Dígito real"], colnames=["Cluster"])
    fig5, ax5 = plt.subplots(figsize=(9, 5))
    sns.heatmap(tabla_cruzada, annot=True, fmt="d", cmap="Blues", ax=ax5)
    st.pyplot(fig5, use_container_width=True)

# --- Tab SVM ----------------------------------------------------------------
with tab_svm:
    st.subheader(f"Clasificación con SVM (kernel RBF) usando {n_components} componentes")

    c1, c2 = st.columns(2)
    c1.metric("Accuracy (test)", f"{resultados['accuracy']*100:.2f}%")
    c2.metric("F1-score macro", f"{resultados['reporte']['macro avg']['f1-score']*100:.2f}%")

    if clasificar or True:
        fila_pca = df_test.iloc[idx_muestra][PCA_COLS].values[:n_components].reshape(1, -1)
        digito_real = int(df_test.iloc[idx_muestra]["digito_real"])
        digito_predicho = int(resultados["svm"].predict(fila_pca)[0])

        col1, col2 = st.columns([1, 2])
        with col1:
            img = reconstruir_imagen(df_test.iloc[idx_muestra][PCA_COLS].values, n_components)
            fig6, ax6 = plt.subplots(figsize=(3, 3))
            ax6.imshow(img, cmap="gray")
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

    st.markdown("#### Reporte de clasificación")
    df_reporte = pd.DataFrame(resultados["reporte"]).transpose().round(3)
    st.dataframe(df_reporte, use_container_width=True)

    st.markdown("#### Matriz de confusión")
    fig7, ax7 = plt.subplots(figsize=(7, 7))
    ConfusionMatrixDisplay.from_predictions(
        y_test, resultados["y_pred"], cmap="Blues", ax=ax7, colorbar=False
    )
    st.pyplot(fig7, use_container_width=True)

# --- Tab Comparación ----------------------------------------------------------
with tab_comp:
    st.subheader("Efecto de la reducción de dimensionalidad (calculado en el notebook)")

    if df_comparacion is None:
        st.warning("No se encontró `outputs/mnist_comparacion_componentes.csv`.")
    else:
        st.dataframe(df_comparacion, use_container_width=True, hide_index=True)

        fig8, axes = plt.subplots(1, 2, figsize=(12, 4.2))
        sns.lineplot(data=df_comparacion, x="n_componentes", y="accuracy", marker="o", ax=axes[0])
        axes[0].axvline(n_components, color="red", linestyle="--")
        axes[0].set_title("Accuracy vs. número de componentes")
        sns.lineplot(data=df_comparacion, x="n_componentes", y="tiempo_entrenamiento_seg",
                     marker="o", color="darkorange", ax=axes[1])
        axes[1].set_title("Tiempo de entrenamiento vs. número de componentes")
        st.pyplot(fig8, use_container_width=True)

    if metadata is not None:
        with st.expander("ℹ️ Metadatos del modelo entrenado en el notebook"):
            st.json(metadata)

st.markdown("---")
st.caption("Proyecto académico · IS-701 Inteligencia Artificial · Campus Comayagua · UNAH")
