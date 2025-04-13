import streamlit as st

# === Page Config ===
st.set_page_config(
    page_title="AI-Powered Food Classification and Calorie Estimation Using the Food-101 Dataset",
    layout="wide"
)

# === Reduce space above title ===
st.markdown("""
    <style>
    .block-container {
        padding-top: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# === App Title ===
st.markdown(
    "<h2 style='text-align: center; color: gray;'>AI-Powered Food Classification and Calorie Estimation Using the Food-101 Dataset</h2>",
    unsafe_allow_html=True
)

# === Download models with spinner ===
with st.spinner("Downloading models, please wait..."):
    from utils.download_models import download_all_models
    download_all_models()

# === Hide Streamlit default menu ===
hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# === Import Main Pages ===
from mainPages import dataset_info, model_comparison, model_training, model_testing, model_evaluation,project_pdf, presentation_pdf

# === App Layout ===
col1, col2 = st.columns([2, 6])

with col1:
    st.markdown("### Main Menu")
    page = st.radio(
        "Go to",
        ["Dataset Information", "Model Training", "Model Comparison", "Model Testing","Model Evaluation"],
        label_visibility="collapsed"
    )
    # , "Project Report", "Project Presentation"

with col2:
    if page == "Dataset Information":
        dataset_info.show()
    elif page == "Model Training":
        model_training.show()
    elif page == "Model Comparison":
        model_comparison.show()
    elif page == "Model Testing":
        model_testing.show()
    elif page == "Model Evaluation":
        model_evaluation.show()        
    # elif page == "Project Report":
    #     project_pdf.show()
    # elif page == "Project Presentation":
    #     presentation_pdf.show()
