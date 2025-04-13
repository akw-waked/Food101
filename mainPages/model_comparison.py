import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import json
import os

def show():
    st.title("📈 Model Training History Comparison")

    # Paths to evaluation files
    paths = {
        "Baseline (lr=0.001, bs=8, freeze=True)": './evaluation/baseline_lr0.001_bs8_freezeTrue/training_history.json',
        "Freeze (lr=0.001, bs=8)": './evaluation/pretrained_freeze_lr0.001_bs8_freezeTrue/training_history.json',
        "Unfreeze (lr=0.001, bs=8)": './evaluation/pretrained_unfreeze_lr0.001_bs8_freezeTrue/training_history.json',
        "Freeze (lr=0.0005, bs=16)": './evaluation/pretrained_freeze_lr0.0005_bs16_freezeTrue/training_history.json',
    }

    # Load histories
    all_dfs = []
    for model_name, path in paths.items():
        if os.path.exists(path):
            with open(path, 'r') as f:
                data = json.load(f)["history"]
                df = pd.DataFrame(data)
                df["Model"] = model_name
                all_dfs.append(df)
        else:
            st.warning(f"File not found: {path}")

    if not all_dfs:
        st.error("No data available to display!")
        return

    combined_df = pd.concat(all_dfs, ignore_index=True)

    # === Model Selection
    selected_models = st.multiselect(
        "Select models to display:",
        options=list(paths.keys()),
        default=list(paths.keys())
    )

    if not selected_models:
        st.warning("Please select at least one model to display the charts and summary.")
        return

    filtered_df = combined_df[combined_df["Model"].isin(selected_models)]

    # === Training Loss
    st.subheader("Training Loss Over Epochs")
    fig_train, ax_train = plt.subplots()
    for model in filtered_df["Model"].unique():
        df_model = filtered_df[filtered_df["Model"] == model]
        ax_train.plot(df_model["epoch"], df_model["train_loss"], marker='o', label=model)
    ax_train.set_xlabel('Epoch')
    ax_train.set_ylabel('Training Loss')
    ax_train.set_title('Training Loss Over Epochs')
    ax_train.legend()
    st.pyplot(fig_train)

    # === Validation Loss
    st.subheader("Validation Loss Over Epochs")
    fig_val_loss, ax_val_loss = plt.subplots()
    for model in filtered_df["Model"].unique():
        df_model = filtered_df[filtered_df["Model"] == model]
        ax_val_loss.plot(df_model["epoch"], df_model["val_loss"], marker='o', label=model)
    ax_val_loss.set_xlabel('Epoch')
    ax_val_loss.set_ylabel('Validation Loss')
    ax_val_loss.set_title('Validation Loss Over Epochs')
    ax_val_loss.legend()
    st.pyplot(fig_val_loss)

    # === Validation Accuracy
    st.subheader("Validation Accuracy Over Epochs")
    fig_val_acc, ax_val_acc = plt.subplots()
    for model in filtered_df["Model"].unique():
        df_model = filtered_df[filtered_df["Model"] == model]
        ax_val_acc.plot(df_model["epoch"], df_model["val_acc"], marker='o', label=model)
    ax_val_acc.set_xlabel('Epoch')
    ax_val_acc.set_ylabel('Validation Accuracy (%)')
    ax_val_acc.set_title('Validation Accuracy Over Epochs')
    ax_val_acc.legend()
    st.pyplot(fig_val_acc)

    # === Summary Table
    st.subheader("📝 Summary Table")
    summary_data = {
        "Model Type": [],
        "Final Training Loss": [],
        "Final Validation Loss": [],
        "Final Validation Accuracy (%)": []
    }

    for model in selected_models:
        df = filtered_df[filtered_df["Model"] == model]
        summary_data["Model Type"].append(model)
        summary_data["Final Training Loss"].append(df["train_loss"].iloc[-1])
        summary_data["Final Validation Loss"].append(df["val_loss"].iloc[-1])
        summary_data["Final Validation Accuracy (%)"].append(df["val_acc"].iloc[-1])

    summary_df = pd.DataFrame(summary_data)
    st.table(summary_df)
