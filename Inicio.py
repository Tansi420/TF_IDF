import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import re
from nltk.stem import SnowballStemmer

# ---------- DISEÑO (la fuente de letra no se modifica) ----------
st.markdown("""
<style>
    /* Fondo general: azul noche con un brillo cálido */
    .stApp {
        background: radial-gradient(circle at top right, #3a1218 0%, #0b1220 45%, #060a14 100%);
        color: #f1f5f9;
    }

    /* Título con franja de color */
    h1 {
        color: #ffffff !important;
        background: linear-gradient(90deg, #d7263d, #ff6b35);
        padding: 0.6rem 1rem;
        border-radius: 12px;
        border-left: 10px solid #ffd23f;
        box-shadow: 0 6px 20px rgba(215, 38, 61, 0.35);
    }

    /* Subtítulos */
    h3 {
        color: #ffd23f !important;
        border-bottom: 2px solid #d7263d;
        padding-bottom: 0.25rem;
    }

    /* Caja de texto y entrada */
    textarea, input {
        background-color: #111a2e !important;
        color: #f8fafc !important;
        border: 1.5px solid #ff6b35 !important;
        border-radius: 10px !important;
    }
    textarea:focus, input:focus {
        border-color: #ffd23f !important;
        box-shadow: 0 0 0 2px rgba(255, 210, 63, 0.35) !important;
    }

    /* Botón principal */
    .stButton > button {
        background: linear-gradient(135deg, #d7263d, #ff6b35);
        color: #ffffff;
        font-weight: bold;
        border: none;
        border-radius: 999px;
        padding: 0.6rem 1.6rem;
        box-shadow: 0 0 18px rgba(215, 38, 61, 0.55);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .stButton > button:hover {
        transform: scale(1.04);
        box-shadow: 0 0 28px rgba(255, 107, 53, 0.8);
        color: #ffffff;
    }

    /* Tablas */
    [data-testid="stDataFrame"] {
        border: 1.5px solid #d7263d;
        border-radius: 10px;
        overflow: hidden;
    }

    /* Alertas */
    [data-testid="stAlert"] {
        border-radius: 10px;
        border-left: 6px solid #ffd23f;
    }
</style>
""", unsafe_allow_html=True)

st.title("Demo de TF-IDF con Preguntas y Respuestas")

st.write("""
Cada línea se trata como un **documento** (puede ser una frase, un párrafo o un texto más largo).  
⚠️ Los documentos y las preguntas deben estar en **inglés**, ya que el análisis está configurado para ese idioma.  

La aplicación aplica normalización y *stemming* para que palabras como *playing* y *play* se consideren equivalentes.
""")

# Ejemplo inicial en inglés
text_input = st.text_area(
    "Escribe tus documentos (uno por línea, en inglés):",
    "The dog barks loudly.\nThe cat meows at night.\nThe dog and the cat play together."
)

question = st.text_input("Escribe una pregunta (en inglés):", "Who is playing?")

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

if st.button("Calcular TF-IDF y buscar respuesta"):
    documents = [d.strip() for d in text_input.split("\n") if d.strip()]
    if len(documents) < 1:
        st.warning("⚠️ Ingresa al menos un documento.")
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
            index=[f"Doc {i+1}" for i in range(len(documents))]
        )

        st.write("### Matriz TF-IDF (stems)")
        st.dataframe(df_tfidf.round(3))

        # Vector de la pregunta
        question_vec = vectorizer.transform([question])

        # Similitud coseno
        similarities = cosine_similarity(question_vec, X).flatten()

        # Documento más parecido
        best_idx = similarities.argmax()
        best_doc = documents[best_idx]
        best_score = similarities[best_idx]

        st.write("### Pregunta y respuesta")
        st.write(f"**Tu pregunta:** {question}")
        st.write(f"**Documento más relevante (Doc {best_idx+1}):** {best_doc}")
        st.write(f"**Puntaje de similitud:** {best_score:.3f}")

        # Mostrar todas las similitudes
        sim_df = pd.DataFrame({
            "Documento": [f"Doc {i+1}" for i in range(len(documents))],
            "Texto": documents,
            "Similitud": similarities
        })
        st.write("### Puntajes de similitud (ordenados)")
        st.dataframe(sim_df.sort_values("Similitud", ascending=False))

        # Mostrar coincidencias de stems
        vocab = vectorizer.get_feature_names_out()
        q_stems = tokenize_and_stem(question)
        matched = [s for s in q_stems if s in vocab and df_tfidf.iloc[best_idx].get(s, 0) > 0]
        st.write("### Stems de la pregunta presentes en el documento elegido:", matched)
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
