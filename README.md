# 🏨 Hotel Booking Cancellation Prediction - ML App (End-to-End)
This project is an end-to-end machine learning application to predict whether a hotel booking will be canceled or not, based on customer and booking details. It's built using Python, Scikit-learn/XGBoost, and Streamlit for deployment.


## 💡 Problem Statement
The goal is to classify hotel bookings as:
Canceled (label = 1)
Not Canceled (label = 0)
by analyzing customer behavior and booking features, this model can help hotels proactively manage inventory and reduce cancellations.

## 🔍 Features
- 📊 Exploratory Data Analysis (EDA)
- 🧹 Data Preprocessing (handling missing data, encoding, scaling)
- 📦 Model Training (XGBoost, Logistic Regression, etc.)
- ✅ Evaluation (Confusion Matrix, ROC-AUC, F1-Score)
- 💾 Model & Scaler Saving (model.pkl, scaler.pkl)
- 🖥️ Final App Deployment using Streamlit
- 🔐 Secrets management using .env or Azure Key Vault

## 📷 Final Streamlit App Output

- Overview Page:  https://praghi-ml-end-to-end-hotelbookingprediction.streamlit.app/
- Prediction Page: https://praghi-ml-end-to-end-hotelbookingprediction.streamlit.app/Prediction


## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/praghi/ML-End-to-End-HotelBookingPrediction-Project.git

2. 🔧 Initialize Git Repository (If Not Already Initialized)
   ```bash
   git init

3. 📋 Check the Status of Your Changes
   ```bash
   git status

4. ✅ Stage Files for Commit
   ```bash
   git add . 

5. 💬 Commit Changes with a Meaningful Message
   ```bash
   git commit -m "Your meaningful commit message here"

6. 🔗 Connect to a Remote Repository (If Not Already Connected)
    ```bash
    git remote add origin <your repo>

7. ⬆️ Push Changes to Remote Repository
    ```bash
    To push to the main branch:
    git push -u origin main

8. ⬇️ Pull Latest Changes Before Pushing (To Avoid Merge Conflicts)
    ```bash
    git pull origin main --rebase

10. 🌿 Create and Switch to a New Branch (If Needed)
   ```bash
    git checkout -b feature-branch-name

🔄 Merge a Branch into Main
Switch to the main branch:
git checkout main

Merge your branch:
git merge feature-branch-name
