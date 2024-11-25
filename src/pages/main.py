import os
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
import PyPDF2
import numpy as np
import json
from sklearn.metrics.pairwise import cosine_similarity

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(
    base_url=os.getenv("LLAMA_BASE_URL"),
    api_key=os.getenv("LLAMA_API_KEY"),
)


# Function for extracting text from PDF
def extract_text_from_pdf(uploaded_file):
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text


# Function to handle uploaded files and generate a class plan
def generate_class_plan_from_files(class_details, uploaded_files):
    """
    Converts PDF and TXT files into strings, limits PDF size to under 1MB,
    and generates a class plan using the OpenAI Llama API.
    """
    documents = []
    for uploaded_file in uploaded_files:
        if uploaded_file.type == "application/pdf":
            # Check size of the file
            if uploaded_file.size > 1 * 1024 * 1024:  # 1MB limit
                st.warning(f"Skipping {uploaded_file.name}: File size exceeds 1MB.")
                continue
            text = extract_text_from_pdf(uploaded_file)
            documents.append(text)
        elif uploaded_file.type == "text/plain":
            text = uploaded_file.read().decode("utf-8")
            documents.append(text)
        else:
            st.warning(f"Unsupported file type: {uploaded_file.type}")

    # Combine all text into a single context
    context = " ".join(documents) if documents else "No reference materials provided."

    print(context)

    # Construct user input
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

    # Use OpenAI client for generating class plans
    with st.spinner("Generating class plan..."):
        completion = client.chat.completions.create(
            model="meta-llama/Llama-3.2-3B-Instruct-Turbo",
            messages=[
                {"role": "system", "content": os.getenv("SYSTEM_PROMPT")},
                {
                    "role": "user",
                    "content": f"Please generate a class plan using the following information: {user_input} in the following language: {class_details.get('language', 'Spanish')}",
                },
            ],
            max_tokens=500,
            temperature=temp,
        )

    return completion.choices[0].message.content


# Main app logic
def load_translations(language):
    if language == "eng":
        with open("./languages/en.json", "r") as f:
            return json.load(f)
    elif language == "esp":
        with open("./languages/es.json", "r") as f:
            return json.load(f)


def app():
    # Language selection
    language = st.session_state.get("language", {})
    translations = load_translations(language)

    # Title and instructions
    st.title(translations["class_plan_generator"])
    st.write(translations["use_saved_details"])

    if "class_details" not in st.session_state:
        st.error(translations["no_class_details"])
        return

    class_details = st.session_state["class_details"]

    st.subheader(translations["class_details_title"])
    st.json(class_details)

    # File upload for additional context
    st.sidebar.header(translations["upload_reference_materials"])
    uploaded_files = st.sidebar.file_uploader(
        translations["upload_documents"], type=["txt", "pdf"], accept_multiple_files=True
    )

    if st.button(translations["generate_class_plan_2"]):
        class_plan = generate_class_plan_from_files(class_details, uploaded_files)
        st.subheader(translations["generated_class_plan"])
        st.write(class_plan)


# Run app
if __name__ == "__main__":
    app()
