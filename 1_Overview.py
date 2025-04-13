import streamlit as st
import pandas as pd
import json
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
  # Import the prediction page function

# Set Streamlit page configuration
st.set_page_config(page_title="Overview", page_icon="📊", layout="wide")



# Paths for model and data
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
METRICS_PATH = os.path.join(PROJECT_ROOT, "metrics.json")
TEST_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "features", "test_features.csv")


def load_metrics():
    """Load model evaluation metrics from JSON."""
    with open(METRICS_PATH, "r") as file:
        return json.load(file)


def load_test_data():
    """Load test data and return features & target."""
    test_data = pd.read_csv(TEST_DATA_PATH)
    X_test = test_data.iloc[:, :-1].values  # Features
    y_test = test_data.iloc[:, -1].values   # Target
    return X_test, y_test

def load_model():
        # Get the parent directory (main directory)
    base_dir =  PROJECT_ROOT # Go up one level
    model_path = os.path.join(base_dir, "model.pkl")  # Use "model.pkl" in the main directory
    scaler_path = os.path.join(base_dir,"scaler.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"🚨 Model file '{model_path}' not found!")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"🚨 Scaler file '{scaler_path}' not found!")

    # Load model
    with open(model_path, 'rb') as f_model:
        xgb_model = pickle.load(f_model)

    # Load scaler
    with open(scaler_path, 'rb') as f_scaler:
        scaler = pickle.load(f_scaler)

    return xgb_model, scaler


def evaluate_model(xgb_model, X_test, y_test):
    """Generate predictions and calculate performance metrics."""
    y_pred = xgb_model.predict(X_test)
    y_pred_proba = xgb_model.predict_proba(X_test)[:, 1]  # Probability scores for positive class
    conf_matrix = confusion_matrix(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    class_report = classification_report(y_test, y_pred, output_dict=True)
    return y_pred, y_pred_proba, conf_matrix, fpr, tpr, roc_auc, class_report


def display_custom_css():
    """Apply custom CSS to style the page."""
    st.markdown(
        """
        <style>
            /* Center title */
            .title { 
                text-align: center;
                font-size: 36px;
                font-weight: bold;
                color: #2E3B55;
            }

            /* Metric cards */
            .metric-container {
                display: flex;
                justify-content: space-around;
                padding: 10px;
            }
            .metric-card {
                background-color: #f8f9fa;
                padding: 20px;
                border-radius: 10px;
                text-align: center;
                box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1);
                min-width: 150px;
                width: 22%;
            }
            .metric-title {
                font-size: 16px;
                font-weight: bold;
                color: #444;
            }
            .metric-value {
                font-size: 22px;
                color: #0056b3;
                font-weight: bold;
            }

            /* Adjust visuals for Confusion Matrix & AUC */
            .visuals-container {
                display: flex;
                justify-content: space-around;
                margin-top: 20px;
            }
            .visual {
                width: 48%;
                text-align: center;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def display_metrics(metrics):
    """Display model performance metrics with custom HTML cards."""
    st.markdown('<h2 class="title">📊 Hotel Booking Model Prediction Dashboard</h2>', unsafe_allow_html=True)

    st.markdown(
        f"""
        <div class="metric-container">
            <div class="metric-card">
                <div class="metric-title">Accuracy</div>
                <div class="metric-value">{metrics['accuracy']:.4f}</div>
            </div>
            <div class="metric-card">
                <div class="metric-title">Precision</div>
                <div class="metric-value">{metrics['precision']:.4f}</div>
            </div>
            <div class="metric-card">
                <div class="metric-title">Recall</div>
                <div class="metric-value">{metrics['recall']:.4f}</div>
            </div>
            <div class="metric-card">
                <div class="metric-title">AUC Score</div>
                <div class="metric-value">{metrics['auc']:.4f}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )




def plot_confusion_matrix(conf_matrix):
    """Generate a confusion matrix with user-friendly labels."""
    
    fig, ax = plt.subplots(figsize=(4, 3))  # Ensure balanced sizing

    # Define class labels for better readability
    class_labels = ["Not Cancelled", "Cancelled"]

    # Plot heatmap with updated labels
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False, 
                xticklabels=class_labels, yticklabels=class_labels, ax=ax)

    ax.set_xlabel("Predicted Labels", fontsize=10)
    ax.set_ylabel("True Labels", fontsize=10)
    
    return fig



def plot_auc_curve(fpr, tpr, roc_auc):
    """Generate a properly aligned AUC Curve."""
    fig, ax = plt.subplots(figsize=(4, 3))  # Matching CM size
    ax.plot(fpr, tpr, color="blue", lw=2, label=f"AUC = {roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], color="grey", linestyle="--")  # Diagonal reference line
    ax.set_xlabel("False Positive Rate", fontsize=10)
    ax.set_ylabel("True Positive Rate", fontsize=10)
    #ax.set_title("ROC Curve", fontsize=12)
    ax.legend(loc="lower right")

    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    return fig


def display_evaluation(conf_matrix, fpr, tpr, roc_auc):
    st.markdown(
        """
        <style>
            .custom-title { 
                text-align: center; 
                font-size: 28px; 
                font-weight: bold; 
            }
            .custom-container {
                display: flex;
                justify-content: center;
                gap: 20px;
            }
        </style>
        """,
        unsafe_allow_html=True
    )
    st.markdown('<div class="spacer"></div>', unsafe_allow_html=True)  # Add space before Confusion Matrix 
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div style="text-align: center; font-size: 16px;"><b>Confusion Matrix</b></div>', unsafe_allow_html=True)
        st.pyplot(plot_confusion_matrix(conf_matrix))  # Ensure this function is available

    with col2:
        st.markdown('<div style="text-align: center; font-size: 16px;"><b>ROC Curve</b></div>', unsafe_allow_html=True)
        st.pyplot(plot_auc_curve(fpr, tpr, roc_auc))  # Ensure this function is available

    # """Display confusion matrix and AUC curve side by side."""
    # st.markdown('<h3 style="text-align: center;">🔍 Model Evaluation</h3>', unsafe_allow_html=True)
    
    # col1, col2 = st.columns(2)
    # with col1:
    #     st.pyplot(plot_confusion_matrix(conf_matrix))
    # with col2:
    #     st.pyplot(plot_auc_curve(fpr, tpr, roc_auc))


def display_classification_report(class_report):
    """Display classification report in tabular format."""
    st.markdown('<h4 style="text-align: center; font-size: 18px;">📝 Classification Report</h4>', unsafe_allow_html=True)
    class_report_df = pd.DataFrame(class_report).transpose()
    st.dataframe(class_report_df.style.format(precision=4))


def main():
    """Main function to load data, evaluate model, and display results."""
    display_custom_css()  # Apply CSS
    metrics = load_metrics()
    X_test, y_test = load_test_data()
    xgb_model, scalar = load_model()
    
    y_pred, y_pred_proba, conf_matrix, fpr, tpr, roc_auc, class_report = evaluate_model(xgb_model, X_test, y_test)

    display_metrics(metrics)
    display_evaluation(conf_matrix, fpr, tpr, roc_auc)
    display_classification_report(class_report)


if __name__ == "__main__":
    main()
