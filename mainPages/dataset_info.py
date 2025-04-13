import streamlit as st

def show():
    st.title("🍽️ Dataset Information: Food-101")

    st.markdown("""
    **Name:** Food-101 Dataset

    **Description:**  
    Food-101 is a large-scale dataset of food images for fine-grained classification. It contains 101 different food categories from all around the world, making it a great benchmark for food recognition tasks.

    **Contents:**
    - 101 food categories
    - 101,000 images total (1,000 images per category)
    - Images split into:
        - Training set: 75,750 images (750 images per class)
        - Test set: 25,250 images (250 images per class)

    **Custom Split:**
    - From the original training set (75,750 images):
        - **Training:** 85% → ~64,387 images
        - **Validation:** 15% → ~11,362 images
    - Test set remains: 25,250 images
                
    **Purpose:**  
    Designed for visual food recognition challenges, food classification, and building AI food recommendation or calorie estimation apps.

    **License:** Publicly available for research and educational use.

    **Download Link:**  
    - [🍔 Food-101 Dataset on Kaggle](https://www.kaggle.com/datasets/dansbecker/food-101)  
    - [📂 Original Food-101 Dataset (ETH Zurich)](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/)
    """)