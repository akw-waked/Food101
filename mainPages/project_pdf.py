import streamlit as st
import base64
import os

def show():
    st.title("📄 Project Report Viewer")

    pdf_path = "./reports/a_waked-FinalProject.pdf"
   
    def display_pdf(base64_pdf):
        pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="800px" type="application/pdf"></iframe>'
        st.markdown(pdf_display, unsafe_allow_html=True)

    try:
        # open PDF
        with open(pdf_path, "rb") as f:
            base64_pdf = base64.b64encode(f.read()).decode('utf-8')
            display_pdf(base64_pdf)

    except FileNotFoundError:
        st.warning(f"PDF not found at: `{pdf_path}`")
        st.info("You can upload the PDF manually to view it below:")

        # Allow manual upload
        uploaded_file = st.file_uploader("Upload PDF", type="pdf")
        if uploaded_file is not None:
            base64_pdf = base64.b64encode(uploaded_file.read()).decode('utf-8')
            display_pdf(base64_pdf)
        else:
            st.error("No PDF uploaded yet. Please upload to view the document.")
