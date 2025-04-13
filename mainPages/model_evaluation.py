import streamlit as st
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
from glob import glob

def show():
    st.title("📊 Model Evaluation Report")

    # === Step 1: Auto-detect model folders ===
    st.subheader("Select Model Evaluation File")
    evaluation_files = glob("./evaluation/*/evaluation.json")
    if not evaluation_files:
        st.error("No evaluation files found in './evaluation'. Please check the path.")
        return

    evaluation_paths = {
        os.path.basename(os.path.dirname(path)).replace("_", " "): path
        for path in evaluation_files
    }

    selected_model = st.selectbox(
        "Choose a model to display:",
        options=list(evaluation_paths.keys())
    )
    evaluation_file = evaluation_paths[selected_model]

    # === Step 2: Load selected evaluation.json ===
    try:
        with open(evaluation_file, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        st.error(f"Evaluation file not found for {selected_model}.")
        return

    # === Step 3: Display overall accuracy ===
    st.subheader("Overall Test Accuracy")
    st.write(f"{data['test_accuracy']:.2f}%")

    # === Step 4: Prepare classification report table ===
    report_data = []
    class_names = []
    for class_name, metrics in data["classification_report"].items():
        if isinstance(metrics, dict):
            class_names.append(class_name)
            report_data.append({
                "Class": class_name.replace("_", " ").title(),
                "Precision": metrics["precision"],
                "Recall": metrics["recall"],
                "F1-Score": metrics["f1-score"],
                "Support": metrics["support"],
                "Raw Class Name": class_name
            })

    df_report = pd.DataFrame(report_data)

    # === Step 5: Display full classification report ===
    st.subheader("Classification Report (All Classes)")
    st.dataframe(
        df_report[["Class", "Precision", "Recall", "F1-Score", "Support"]].style.format({
            "Precision": "{:.2f}",
            "Recall": "{:.2f}",
            "F1-Score": "{:.2f}",
            "Support": "{:.0f}"
        }),
        use_container_width=True
    )

    # === Step 6: Class selection for confusion matrix ===
    st.markdown("#### Select classes to display in Confusion Matrix:")
    available_classes = df_report.sort_values(by="F1-Score", ascending=False)["Class"].tolist()

    selected_classes = st.multiselect(
        "Select classes to display (default: Top 5 by F1-Score):",
        options=available_classes,
        default=available_classes[:5]
    )

    df_selected = df_report[df_report["Class"].isin(selected_classes)]

    # === Step 7: Confusion Matrix ===
    if "confusion_matrix" in data and not df_selected.empty:
        st.subheader("Confusion Matrix (Selected Classes Only)")

        cm = np.array(data["confusion_matrix"])

        selected_raw_classes = df_selected["Raw Class Name"].tolist()
        all_classes = list(data["classification_report"].keys())
        selected_indices = [all_classes.index(cls) for cls in selected_raw_classes]

        cm_selected = cm[np.ix_(selected_indices, selected_indices)]

        fig, ax = plt.subplots(figsize=(3, 3))
        sns.heatmap(cm_selected, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=df_selected["Class"], yticklabels=df_selected["Class"])
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        st.pyplot(fig)


    else:
        st.info("No confusion matrix found in the evaluation file or no classes selected.")
    # Add interpretation text under confusion matrix
    st.markdown("""
    > **📝 Interpretation:**  
    > The confusion matrix shows how well the model distinguishes between different classes.  
    > Misclassifications typically occur between classes that have similar visual characteristics or belong to similar categories.  
    > Improving dataset quality, adding diverse images, and applying targeted data augmentation can help the model better differentiate between challenging classes.
    """)
