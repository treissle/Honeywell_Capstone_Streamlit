import streamlit as st
from io import BytesIO
from utils import DocumentProcessor, classify_text
from pdf2image import convert_from_bytes
import torch
torch.classes.__path__ = []



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

    st.divider()
   
    if input_method == "Upload File(s)" and uploaded_files:
        for uploaded_file in uploaded_files:
            with st.expander(uploaded_file.name):
                file_bytes = uploaded_file.getvalue()
                uploaded_file.seek(0)

                try:
                    processed_document = DocumentProcessor(BytesIO(file_bytes))
                
                except Exception as e:
                    st.error(f"Error processing file {uploaded_file.name}: {e}")
                    continue
    

    elif input_method == "Provide Text Portion" and text_portion:
        # text_portion_classification = classify_text(text_portion)
        text_portion_classification = False

        if text_portion_classification:
            st.badge("CUI", icon="⚠️", color="red")
        else:
            st.badge("No CUI", icon="✅", color="green")
        
        st.code(text_portion, language="None")


if __name__ == "__main__":
    main()
