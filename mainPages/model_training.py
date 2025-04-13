import streamlit as st
import pandas as pd

def show():
    st.header("🧩 Model Architecture")
    st.markdown("""
    - **ResNet18**
    """)

    # Model Types
    st.header("Model Types")
    st.markdown("""
    - **Baseline**
    - **Pretrained (Frozen layers)**
    - **Pretrained (Fine-tuned)**
    """)

    # Data Augmentation
    st.header("Data Augmentation")
    st.markdown("""
    - **Resize:** (224x224)
    - **RandomHorizontalFlip**
    - **RandomRotation:** 10 degrees
    - **ColorJitter:** brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
    - **ToTensor**
    - **Normalize**
    """)

    # Hyperparameters
    st.header("Hyperparameters")
    st.markdown("""
    - **Loss Function:** CrossEntropyLoss
    - **Optimizer:** Adam
    - **Learning Rate:** 0.001
    - **Epochs:** 10
    - **Patience:** 2
    - **Device:** GPU
    """)

    # Training Results Table
    st.header("Training Results")

    data = {
        "Model Type": ["Baseline", "Pretrained Freeze", "Pretrained Unfreeze"],
        "Training Loss": [1.6, 0.91, 0.98],
        "Validation Loss": [1.8, 1.18, 1.46],
        "Validation Accuracy": ["52.3%", "69.13%", "63.68%"],
        "Epochs": [10, 8, 10],
        "Early Stop": ["No", "Yes", "No"]
    }

    df = pd.DataFrame(data)
    st.table(df)

    # Notes
    st.header("Notes")
    st.markdown("""
    - **Early stopping** helped prevent overfitting in pretrained freeze model.
    - **Pretrained models** achieved higher accuracy than baseline.
    """)
