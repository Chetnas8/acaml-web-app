[![Streamlit](https://img.shields.io/badge/Deployed%20on-Streamlit-brightgreen)](https://your-streamlit-app-link.streamlit.app)

# 📊 ACAML: Adaptive Constraint-Aware AutoML Web Application

Welcome to **ACAML**, an innovative web-based AutoML application built with Streamlit. This project demonstrates my ability to design, implement, and deploy real-world machine learning solutions with user-friendly interfaces and powerful automation features.

---

## 🚀 Project Overview

ACAML empowers users to:
- Upload their own datasets in CSV format
- Automatically detect the appropriate machine learning task (classification or regression)
- Train the best model using FLAML with time-budgeted optimization
- Display key performance metrics (Accuracy or R² Score) clearly and intuitively
- Visualize model interpretability through SHAP feature importance plots

👉 **[Live Demo of the App](https://acaml-web-app-aepx9uhzknncmoyx7hbszc.streamlit.app/)**

This app is designed to democratize machine learning for users of all skill levels, making it easy to explore, train, and interpret models interactively.

---

## 🛠️ Key Features

- **Dynamic Task Detection**: Automatically classifies datasets as regression or classification based on target variable type and cardinality.
- **Automated Model Selection**: Leverages FLAML to choose the best model and optimize hyperparameters within a specified time budget.
- **User-Centric Interface**: Clean, tabbed layout with separate sections for configuration, results, and explainability.
- **Model Interpretability**: Integrated SHAP visualizations to understand feature importance and build trust in the model.
- **Flexible Deployment**: Fully containerized and deployable on Streamlit Cloud or locally.

---

## 🔍 Technologies Used

- **Python**
- **Streamlit**
- **FLAML**
- **SHAP**
- **scikit-learn**
- **pandas, numpy, matplotlib**

---

## 📸 Screenshots

Screenshots of key app pages (Upload, Results, Explainability) are included in the `screenshots/` folder. They showcase the intuitive user interface and visualizations.

---

## 📈 Why This Matters

As a Data Analyst/Machine Learning Engineer, I built ACAML to showcase my skills in:
- Designing user-friendly data science tools
- Implementing end-to-end ML workflows from preprocessing to model interpretability
- Deploying and maintaining ML applications in real-world scenarios

This project demonstrates my ability to bridge the gap between technical implementation and user experience, ensuring that complex machine learning processes are accessible and understandable to non-technical users.

---

## 📝 Getting Started

Clone the repo and run locally:

```bash
git clone https://github.com/Chetnas8/acaml-web-app.git
cd acaml-web-app
pip install -r requirements.txt
streamlit run app.py
