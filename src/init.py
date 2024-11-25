import streamlit as st
import json
import os

# Function to load translations based on language preference
def load_translations(language):
    if language == 'eng':
        with open('./languages/en.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    elif language == 'esp':
        with open('./languages/es.json', 'r', encoding='utf-8') as f:
            return json.load(f)

# Set page config
st.set_page_config(page_title="AcademIA", page_icon="🎓")

# Initialize session state for language if not already set
if 'language' not in st.session_state:
    st.session_state.language = 'esp'

# Language selection in sidebar
def change_language():
    st.session_state.language = st.sidebar.selectbox(
        'Select Language', ['esp', 'eng'], index=['esp', 'eng'].index(st.session_state.language)
    )
def change_language_sp():
    st.session_state.language = st.sidebar.selectbox(
        'Seleccionar Idioma', ['esp', 'eng'], index=['esp', 'eng'].index(st.session_state.language)
    )

# Call the language change function
if st.session_state.language == 'esp':
    change_language_sp()
else:
    change_language()

# Load the selected language's translations
translations = load_translations(st.session_state.language)

# Display text using the translations
st.title(translations['welcome_message'])
col1, col2, col3 = st.columns([1, 6, 1])  # Adjust these numbers to control the spacing
with col2:
    st.image("./assets/academ-ia-2.png", caption="AcademIA logo", width=250, clamp=True)
st.write(f"""
    {translations['instructions']}
    - {translations['start_instructions']}
    - {translations['generate_class_plan']}
    """)

# Footer
st.write(translations['footer'])
