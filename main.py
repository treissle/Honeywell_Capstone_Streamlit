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
        Upload documents or provide text portions to conduct CUI classification
        """
    )

    # Create two columns for input method selection and input field
    col1, col2 = st.columns(2)

    with col1:
        input_method = st.radio(
            "Choose input method:",
            ("Upload File(s)", "Provide Folder Path", "Provide Text Portion"),
            captions=("Classify one or multiple files", "Classify an entire folder of files", "Classify a text portion")
        )

    with col2:
        if input_method == "Upload File(s)":
            uploaded_files = st.file_uploader("Upload Documents", type=["pdf"], accept_multiple_files=True)
        elif input_method == "Provide Folder Path":
            folder_path = st.text_input("Provide folder path and press enter to classify")
        else:
            text_portion = st.text_area("Provide text portion to be classified and press **CTRL + ENTER**")

    st.divider()
   
    if input_method == "Upload File(s)" and uploaded_files:
        for uploaded_file in uploaded_files:
            # try:
            file_bytes = uploaded_file.read()
            # uploaded_file.seek(0)

            processed_document = DocumentProcessor(file=BytesIO(file_bytes))
            pages = processed_document.draw_bounding_boxes()
            
            # except Exception as e:
            #     st.error(f"Error processing file {uploaded_file.name}: {e}")
            #     continue
            
            

            icon, expander_text = ("⚠️", "Likely contains CUI") if processed_document.file_CUI_classification else ("✅", "Likely does not contain CUI")

            
            with st.expander(f"{uploaded_file.name} - {expander_text}", icon=icon):
                captions = [f"Page {i+1}" for i in range(len(pages))]
                with st.container(height=450, border=True):
                    st.markdown(':red[Red] = CUI ● :green[Green] = No CUI')
                    st.image(pages, caption=captions)
    

    elif input_method == "Provide Text Portion" and text_portion:
        text_portion_classification = classify_text(text_portion)
        # text_portion_classification = True

        if text_portion_classification:
            st.badge("CUI", icon="⚠️", color="red")
            st.subheader("This text portion likely contains CUI")
        else:
            st.badge("No CUI", icon="✅", color="green")
            st.subheader("This text portion likely does not contain CUI")
        
        st.code(text_portion, language=None)


if __name__ == "__main__":
    main()
