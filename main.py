import streamlit as st
from io import BytesIO
from utils import DocumentProcessor, classify_text
from pdf2image import convert_from_bytes


def main():
    st.set_page_config(page_title="CUI Classification", page_icon='🔍', layout='wide')
    st.title("🔍 File CUI Classification")
    st.logo("Honeywell_logo.png")


    st.write(
        """
        Upload one or more documents. Each document will be processed and displayed 
        within its own expander. Pages will show bounding boxes in **red** if they contain CUI,
        or **green** if they do not.
        """
    )


    # Create two columns for input method selection and input field
    col1, col2 = st.columns(2)

    with col1:
        input_method = st.radio(
            "Choose input method:",
            ("Upload File(s)", "Provide Folder Path", "Provide Text Portion"),
            captions=("Classify one or multiple files", "Classify an entire folder", "Classify a text portion")
        )

    with col2:
        if input_method == "Upload File(s)":
            uploaded_files = st.file_uploader("Upload Documents", type=["pdf", "docx"], accept_multiple_files=True)
        elif input_method == "Provide Folder Path":
            folder_path = st.text_input("Provide folder path and press enter to classify")
        else:
            text_portion = st.text_area("Provide text portion to be classified")
