import os
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
import PyPDF2
import numpy as np
import json
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image

# Cargar variables de entorno
load_dotenv()

# Inicializar cliente de OpenAI
client = OpenAI(
    base_url=os.getenv("LLAMA_BASE_URL"),
    api_key=os.getenv("LLAMA_API_KEY"),
)

# Función para cargar CSS personalizado
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Aplicar el CSS desde el archivo
local_css(os.path.join("assets", "styles.css"))

# Función para cargar modelo GloVe
def load_glove_model(glove_file=os.path.join("..", "data", "glove.6B.50d.txt")):
    """Carga el modelo GloVe desde un archivo."""
    embeddings_index = {}
    with open(glove_file, 'r', encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    return embeddings_index

# Inicializar modelo GloVe
glove_model = load_glove_model()

# Función para generar embeddings usando GloVe
def get_embedding_glove(text, embeddings_index):
    """Genera un embedding GloVe para un texto dado."""
    words = text.split()
    embeddings = [embeddings_index.get(word, np.zeros(50)) for word in words]
    if embeddings:
        return np.mean(embeddings, axis=0)
    else:
        return np.zeros(50)

# Función para extraer texto de archivos PDF
def extract_text_from_pdf(uploaded_file):
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Función para dividir texto en chunks
def chunk_text(text, chunk_size=50):
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

# Función para procesar archivos subidos
def process_uploaded_files(uploaded_files):
    document_chunks = []
    for uploaded_file in uploaded_files:
        if uploaded_file.type == "application/pdf":
            text = extract_text_from_pdf(uploaded_file)
            chunks = chunk_text(text)
            document_chunks.extend(chunks)
        else:
            text = uploaded_file.read().decode("utf-8")
            chunks = chunk_text(text)
            document_chunks.extend(chunks)
    return document_chunks

# Función para generar embeddings
def generate_embeddings(chunks):
    embeddings = []
    for chunk in chunks:
        embedding = get_embedding_glove(chunk, glove_model)
        embeddings.append(embedding)
    return embeddings

# Función para encontrar chunks relevantes
def find_relevant_chunks(embeddings, query_embedding, chunks, top_k=3):
    similarities = cosine_similarity([query_embedding], embeddings)[0]
    top_indices = similarities.argsort()[-top_k:][::-1]
    return [chunks[i] for i in top_indices]

# Función para generar plan de clase
def generate_class_plan(class_details, embeddings, document_chunks):
    with st.spinner("Generando el plan de clase..."):
        if embeddings:
            query_embedding = get_embedding_glove("class topic", glove_model)
            relevant_chunks = find_relevant_chunks(embeddings, query_embedding, document_chunks)
            context = " ".join(relevant_chunks)
        else:
            context = "No hay materiales de referencia."

        user_input = (
            f"Class topic: {class_details['subject']}, Number of students: {class_details['num_students']}, "
            f"Time available: {class_details['time_available']} minutes, Class level: {class_details['level']}, "
            f"Modality: {class_details['modality']}, Purpose: {class_details['purpose']}, "
            f"Language: {class_details.get('language', 'Spanish')}, Special instructions: {class_details.get('instructions', 'None')}, "
            f"Reference materials context: {context}"
        )

        temp = 0.7
        if class_details["field"] == "STEM":
            temp = 0.2
        elif class_details["field"] == "Social Sciences":
            temp = 0.65
        elif class_details["field"] == "Liberal Arts":
            temp = 0.8

        completion = client.chat.completions.create(
            model="meta-llama/Llama-3.2-3B-Instruct-Turbo",
            messages=[
                {"role": "system", "content": os.getenv("SYSTEM_PROMPT")},
                {"role": "user", "content": f"Please generate a class plan using the following information: {user_input} in the following language: {class_details.get('language', 'Spanish')}" },
            ],
            temperature=temp,
        )

        return completion.choices[0].message.content

# Función para cargar traducciones
def load_translations(language):
    if language == 'eng':
        with open('../languages/en.json', 'r') as f:
            return json.load(f)
    elif language == 'esp':
        with open('../languages/es.json', 'r') as f:
            return json.load(f)

# Función principal
def app():
    # Cargar logo
    logo_path = os.path.join("assets", "academ-ia-2.png")
    logo = Image.open(logo_path)
    st.image(logo, width=200)

    # Título principal
    st.title("🎓 Bienvenidos a AcademIA")

    # Selección de idioma
    language = st.session_state.get('language', 'esp')
    translations = load_translations(language)
    st.sidebar.header(translations['options'])
    st.sidebar.selectbox(
        translations['select_language'], options=["esp", "eng"], key="language", index=0
    )

    # Mostrar detalles de la clase
    if "class_details" not in st.session_state:
        st.error(translations['no_class_details'])
        return

    class_details = st.session_state["class_details"]
    st.subheader(translations['class_details_title'])
    st.json(class_details)

    # Subida de archivos
    st.sidebar.header(translations['upload_reference_materials'])
    uploaded_files = st.sidebar.file_uploader(
        translations['upload_documents'], type=["txt", "pdf"], accept_multiple_files=True
    )

    # Botón para generar plan de clase
    if st.button(translations['generate_class_plan']):
        document_chunks = process_uploaded_files(uploaded_files)
        embeddings = generate_embeddings(document_chunks) if document_chunks else []
        class_plan = generate_class_plan(class_details, embeddings, document_chunks)

        st.success(translations['success_message'])
        st.subheader(translations['generated_class_plan'])
        st.write(class_plan)

# Ejecutar aplicación
if __name__ == "__main__":
    app()
