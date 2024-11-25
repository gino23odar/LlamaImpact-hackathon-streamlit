import streamlit as st
import json
import os

# Function to load translations based on the selected language
def load_translations(language):
    path_to_json_en = os.path.join('.', 'languages', 'en.json')
    path_to_json_es = os.path.join('.', 'languages', 'es.json')
    if language == 'eng':
        with open(path_to_json_en, 'r') as f:
            return json.load(f)
    elif language == 'esp':
        with open(path_to_json_es, 'r') as f:
            return json.load(f)
        

def app():
    # Language selection
    language = st.session_state.get('language', {})
    translations = load_translations(language)

    # Display translations for the UI elements
    st.title(translations['settings_title'])
    st.write(translations['settings_instructions'])

    # Required Fields
    subject = st.text_input(translations['subject'])
    field = st.selectbox(
        translations['field_type'], translations['field_options'] 
    )
    num_students = st.number_input(translations['num_students'], min_value=1, step=1)
    time_available = st.slider(translations['class_duration'], 15, 180, 60)

    # Dropdowns for options
    level = st.selectbox(
        translations['class_type'], translations['class_type_options']
    )
    
    if st.session_state.language == 'esp':
        modality = st.radio(translations['modality'], ["Virtual", "Presencial"])
    else:
        modality = st.radio(translations['modality'], ["Virtual", "Presential"])
    purpose = st.selectbox(
        translations['class_purpose'], translations['class_purpose_options']
    )

    # Optional Fields
    language_input = st.text_input(translations['language_input'], value="Espa\u00f1ol")
    instructions = st.text_area(translations['special_instructions'])

    # Save details to session state
    if st.button(translations['save_button']):
        st.session_state.class_details = {
            "subject": subject,
            "field": field,
            "num_students": num_students,
            "time_available": time_available,
            "level": level,
            "modality": modality,
            "purpose": purpose,
            "language": language_input,
            "instructions": instructions,
        }
        st.success(translations['success_message'])

# Run app
if __name__ == "__main__":
    app()

