import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import re
from nltk.stem import SnowballStemmer

# ---------- Estética (no se modifica la fuente de letra) ----------
st.set_page_config(page_title="Botiquín Digital", page_icon="🚑", layout="centered")

st.markdown("""
<style>
    /* Fondo general */
    .stApp {
        background: linear-gradient(160deg, #fff5f5 0%, #ffffff 45%, #eaf6ff 100%);
    }

    /* Franja superior tipo ambulancia */
    header[data-testid="stHeader"] {
        background: linear-gradient(90deg, #c62828, #ef5350, #c62828);
    }

    /* Título principal */
    h1 {
        color: #b71c1c !important;
        border-bottom: 4px solid #c62828;
        padding-bottom: 0.4rem;
    }

    /* Subtítulos */
    h3 {
        color: #1565c0 !important;
        border-left: 6px solid #c62828;
        padding-left: 0.6rem;
        margin-top: 1.5rem;
    }

    /* Cajas de texto */
    .stTextArea textarea, .stTextInput input {
        background-color: #ffffff !important;
        border: 2px solid #ef9a9a !important;
        border-radius: 12px !important;
    }
    .stTextArea textarea:focus, .stTextInput input:focus {
        border-color: #c62828 !important;
        box-shadow: 0 0 0 2px rgba(198, 40, 40, 0.2) !important;
    }

    /* Botón de acción */
    .stButton > button {
        background: linear-gradient(135deg, #c62828, #e53935);
        color: white;
        border: none;
        border-radius: 999px;
        padding: 0.7rem 1.8rem;
        font-weight: 700;
        letter-spacing: 0.5px;
        box-shadow: 0 4px 12px rgba(198, 40, 40, 0.35);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(198, 40, 40, 0.5);
        color: white;
    }

    /* Tablas */
    div[data-testid="stDataFrame"] {
        border: 2px solid #90caf9;
        border-radius: 12px;
        overflow: hidden;
    }

    /* Alertas */
    div[data-testid="stAlert"] {
        border-radius: 12px;
    }
</style>
""", unsafe_allow_html=True)

# ---------- Textos de la herramienta ----------
st.title("🚑 Botiquín Digital: Buscador de Protocolos de Primeros Auxilios")

st.write("""
Cada línea se trata como un **protocolo** (puede ser una instrucción, un párrafo o un procedimiento más largo de tu manual de emergencias).  
⚠️ Los protocolos y la descripción de la emergencia deben estar en **inglés**, ya que el análisis está configurado para ese idioma.  

Describe lo que está pasando y la aplicación te mostrará el protocolo más parecido. Aplica normalización y *stemming* para que palabras como *bleeding* y *bleed* se consideren equivalentes.  
🩺 Herramienta con fines educativos: ante una emergencia real, llama siempre a los servicios de emergencia.
""")

# Ejemplo inicial en inglés
text_input = st.text_area(
    "Escribe los protocolos de tu manual (uno por línea, en inglés):",
    "Apply firm pressure to the wound to stop the bleeding.\nCool the burn under running water for ten minutes.\nCall emergency services if the person is not breathing."
)

question = st.text_input("Describe la emergencia (en inglés):", "What should I do if someone is bleeding?")

# Inicializar stemmer para inglés
stemmer = SnowballStemmer("english")

def tokenize_and_stem(text: str):
    # Pasar a minúsculas
    text = text.lower()
    # Eliminar caracteres no alfabéticos
    text = re.sub(r'[^a-z\s]', ' ', text)
    # Tokenizar (palabras con longitud > 1)
    tokens = [t for t in text.split() if len(t) > 1]
    # Aplicar stemming
    stems = [stemmer.stem(t) for t in tokens]
    return stems

if st.button("🚨 Analizar emergencia y buscar protocolo"):
    documents = [d.strip() for d in text_input.split("\n") if d.strip()]
    if len(documents) < 1:
        st.warning("⚠️ Ingresa al menos un protocolo.")
    else:
        # Vectorizador con stemming
        vectorizer = TfidfVectorizer(
            tokenizer=tokenize_and_stem,
            stop_words="english",
            token_pattern=None
        )

        # Ajustar con documentos
        X = vectorizer.fit_transform(documents)

        # Mostrar matriz TF-IDF
        df_tfidf = pd.DataFrame(
            X.toarray(),
            columns=vectorizer.get_feature_names_out(),
            index=[f"Protocolo {i+1}" for i in range(len(documents))]
        )

        st.write("### 🧬 Matriz de términos clave del manual (TF-IDF)")
        st.dataframe(df_tfidf.round(3))

        # Vector de la pregunta
        question_vec = vectorizer.transform([question])

        # Similitud coseno
        similarities = cosine_similarity(question_vec, X).flatten()

        # Documento más parecido
        best_idx = similarities.argmax()
        best_doc = documents[best_idx]
        best_score = similarities[best_idx]

        st.write("### 🆘 Emergencia y protocolo recomendado")
        st.write(f"**Tu emergencia:** {question}")
        st.write(f"**Protocolo más relevante (Protocolo {best_idx+1}):** {best_doc}")
        st.write(f"**Nivel de coincidencia:** {best_score:.3f}")

        # Mostrar todas las similitudes
        sim_df = pd.DataFrame({
            "Documento": [f"Protocolo {i+1}" for i in range(len(documents))],
            "Texto": documents,
            "Similitud": similarities
        })
        st.write("### 📋 Coincidencia con cada protocolo (ordenados)")
        st.dataframe(sim_df.sort_values("Similitud", ascending=False))

        # Mostrar coincidencias de stems
        vocab = vectorizer.get_feature_names_out()
        q_stems = tokenize_and_stem(question)
        matched = [s for s in q_stems if s in vocab and df_tfidf.iloc[best_idx].get(s, 0) > 0]
        st.write("### 🔎 Palabras clave de la emergencia presentes en el protocolo elegido:", matched)
