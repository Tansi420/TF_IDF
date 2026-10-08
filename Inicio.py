import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import re
from nltk.stem import SnowballStemmer

# ---------- DISEÑO LLAMATIVO (solo CSS; la fuente de letra no se modifica) ----------
st.markdown("""
<style>
    /* ===== Fondo animado tipo aurora ===== */
    .stApp {
        background: linear-gradient(-45deg, #1a0b3b, #0f2a6b, #6a11cb, #ff0080, #00c6ff, #1a0b3b);
        background-size: 400% 400%;
        animation: aurora 18s ease infinite;
        color: #ffffff;
    }
    @keyframes aurora {
        0%   { background-position: 0% 50%; }
        50%  { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    /* Cabecera transparente */
    [data-testid="stHeader"] { background: transparent; }

    /* ===== Tarjeta de cristal (glassmorphism) ===== */
    .block-container {
        background: rgba(255, 255, 255, 0.08);
        backdrop-filter: blur(18px);
        -webkit-backdrop-filter: blur(18px);
        border: 1.5px solid rgba(255, 255, 255, 0.28);
        border-radius: 28px;
        box-shadow: 0 0 60px rgba(255, 0, 128, 0.35), 0 0 120px rgba(0, 198, 255, 0.25);
        padding: 2.5rem 2.5rem 3rem 2.5rem !important;
        margin-top: 1.5rem;
        margin-bottom: 1.5rem;
    }

    /* ===== Título con degradado neón brillante ===== */
    h1 {
        background: linear-gradient(90deg, #00f5ff, #ff00e5, #ffe600, #00f5ff);
        background-size: 300% 100%;
        -webkit-background-clip: text;
        background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: shine 6s linear infinite;
        font-weight: 900 !important;
        letter-spacing: 1px;
        text-shadow: 0 0 30px rgba(255, 0, 229, 0.4);
        padding-bottom: 0.6rem !important;
        border-bottom: 4px solid transparent;
        border-image: linear-gradient(90deg, #00f5ff, #ff00e5, #ffe600) 1;
        margin-bottom: 1rem;
    }
    @keyframes shine {
        0%   { background-position: 0% 50%; }
        100% { background-position: 300% 50%; }
    }

    /* ===== Subtítulos con barra de color ===== */
    h3 {
        color: #ffffff !important;
        font-weight: 800 !important;
        padding: 0.45rem 1rem !important;
        margin-top: 1.5rem !important;
        border-left: 8px solid #ffe600;
        border-radius: 0 14px 14px 0;
        background: linear-gradient(90deg, rgba(255, 0, 229, 0.45), rgba(0, 245, 255, 0.05));
        text-shadow: 0 0 12px rgba(0, 245, 255, 0.7);
    }

    /* Texto general y negritas resaltadas */
    p, li, label, .stMarkdown { color: #f5f3ff; }
    strong { color: #ffe600; text-shadow: 0 0 8px rgba(255, 230, 0, 0.45); }
    em { color: #7df9ff; }

    /* Etiquetas de los campos */
    [data-testid="stWidgetLabel"] p {
        color: #7df9ff !important;
        font-weight: 700;
        font-size: 1.02rem;
        text-shadow: 0 0 10px rgba(0, 245, 255, 0.5);
    }

    /* ===== Campos de texto con brillo neón ===== */
    textarea, input {
        background: rgba(10, 5, 35, 0.75) !important;
        color: #ffffff !important;
        border: 2px solid #00f5ff !important;
        border-radius: 16px !important;
        box-shadow: 0 0 14px rgba(0, 245, 255, 0.45), inset 0 0 12px rgba(0, 245, 255, 0.12);
        transition: all 0.25s ease;
    }
    textarea:focus, input:focus {
        border-color: #ff00e5 !important;
        box-shadow: 0 0 26px rgba(255, 0, 229, 0.8), inset 0 0 14px rgba(255, 0, 229, 0.2) !important;
        transform: translateY(-2px);
    }
    [data-baseweb="textarea"], [data-baseweb="input"], [data-baseweb="base-input"] {
        background: transparent !important;
        border-radius: 16px !important;
    }

    /* ===== Botón gigante pulsante ===== */
    .stButton > button {
        background: linear-gradient(135deg, #ff00e5, #7a00ff, #00c6ff);
        background-size: 200% 200%;
        animation: aurora 6s ease infinite, pulse 2.2s ease-in-out infinite;
        color: #ffffff !important;
        font-weight: 900;
        font-size: 1.1rem;
        letter-spacing: 0.5px;
        border: 2px solid rgba(255, 255, 255, 0.7);
        border-radius: 999px;
        padding: 0.8rem 2.2rem;
        transition: transform 0.2s ease;
    }
    .stButton > button:hover {
        transform: scale(1.08) rotate(-1deg);
        border-color: #ffe600;
        color: #ffffff !important;
    }
    .stButton > button:active { transform: scale(0.97); }
    @keyframes pulse {
        0%, 100% { box-shadow: 0 0 18px rgba(255, 0, 229, 0.7), 0 0 40px rgba(0, 198, 255, 0.4); }
        50%      { box-shadow: 0 0 34px rgba(255, 230, 0, 0.8), 0 0 70px rgba(255, 0, 229, 0.6); }
    }

    /* ===== Tablas ===== */
    [data-testid="stDataFrame"] {
        border: 2px solid #ff00e5;
        border-radius: 18px;
        overflow: hidden;
        box-shadow: 0 0 24px rgba(255, 0, 229, 0.55), 0 0 50px rgba(0, 245, 255, 0.25);
    }

    /* ===== Alertas ===== */
    [data-testid="stAlert"] {
        border-radius: 16px;
        border: 2px solid #ffe600;
        box-shadow: 0 0 20px rgba(255, 230, 0, 0.5);
    }

    /* ===== Barra de scroll de neón ===== */
    ::-webkit-scrollbar { width: 12px; }
    ::-webkit-scrollbar-track { background: rgba(0, 0, 0, 0.3); }
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(#ff00e5, #00f5ff);
        border-radius: 10px;
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
