import streamlit as st
import base64

def show():
    st.title("📄 Project Summary")

    # Local file path
    pdf_path = "./reports/a_waked-Presentation.pdf"

    # Google Drive direct link
    pdf_url = "https://drive.google.com/uc?export=view&id=1U6UO8bG2qwqEvgpFW4h3xwlePXfU2NrM"

    def display_pdf_from_url(url):
        st.markdown(
            f'<iframe src="{url}" width="100%" height="800px"></iframe>',
            unsafe_allow_html=True
        )

    def display_pdf_from_base64(base64_pdf):
        st.markdown(
            f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="800px"></iframe>',
            unsafe_allow_html=True
        )

    try:
        with open(pdf_path, "rb") as f:
            base64_pdf = base64.b64encode(f.read()).decode('utf-8')
            display_pdf_from_base64(base64_pdf)

    except FileNotFoundError:
        st.warning(f"Local PDF not found at: `{pdf_path}`")

        if pdf_url:
            st.info("Loading PDF from external link...")
            display_pdf_from_url(pdf_url)
        else:
            st.info("Upload the PDF manually to view it below:")
            uploaded_file = st.file_uploader("Upload PDF", type="pdf")
            if uploaded_file is not None:
                base64_pdf = base64.b64encode(uploaded_file.read()).decode('utf-8')
                display_pdf_from_base64(base64_pdf)
            else:
                st.error("No PDF uploaded yet. Please upload to view the document.")
