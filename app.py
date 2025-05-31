import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
from flaml import AutoML
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="ACAML : Adaptive Constraint-Aware AutoML")
st.title("📊 ACAML : Adaptive Constraint-Aware AutoML")

tabs = st.tabs(["Upload & Configure", "Results", "Explainability"])

with tabs[0]:
    st.subheader("🔍 Upload & Configure")
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write(df.head())

        # Encode object (string) columns to numeric codes
        df = df.apply(lambda col: col.astype('category').cat.codes if col.dtypes == 'object' and col.name != 'target' else col)

        target_column = st.selectbox("🎯 Select your target column", df.columns)
        time_budget = st.slider("⏱️ Time Budget (seconds)", 10, 300, 60)
        interpret_only = st.checkbox("🧠 Prefer interpretable models only", value=False)

        if st.button("🚀 Run ACAML"):
            if df[target_column].dtype == object:
                try:
                    df[target_column] = df[target_column].str.replace(',', '').astype(float)
                except:
                    pass

            if pd.api.types.is_numeric_dtype(df[target_column]) and df[target_column].nunique() > 10:
                task_type = "regression"
            else:
                task_type = "classification"

            estimator_list = ["lrl1", "lrl2"] if interpret_only else 'auto'

            automl_settings = {
                "time_budget": time_budget,
                "metric": "r2" if task_type == "regression" else "accuracy",
                "task": task_type,
                "log_file_name": "acaml_ui.log",
                "estimator_list": estimator_list
            }

            X = df.drop(columns=[target_column])
            y = df[target_column]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            automl = AutoML()
            try:
                automl.fit(X_train=X_train, y_train=y_train, **automl_settings)
                y_pred = automl.predict(X_test)
                acc = accuracy_score(y_test, y_pred) if task_type == 'classification' else automl.score(X_test, y_test)

                st.session_state['X_train'] = X_train
                st.session_state['X_test'] = X_test
                st.session_state['automl'] = automl
                st.session_state['accuracy'] = acc
                st.session_state['task_type'] = task_type
                st.success("✅ Training complete!")

            except Exception as e:
                st.error(f"Training failed: {e}")

with tabs[1]:
    st.subheader("📈 Model Performance")
    if 'accuracy' in st.session_state:
        score = st.session_state['accuracy']
        task_label = "Accuracy" if st.session_state['task_type'] == 'classification' else "R2 Score"
        # Display the main metric
        st.markdown(f"""
            <div style='text-align: center; font-size: 48px; font-weight: bold; color: #2E86AB;'>
                {task_label}: {score*100:.2f}%
            </div>
        """, unsafe_allow_html=True)

        # Get and display the best model's name in big style
        model_name = type(st.session_state['automl'].model).__name__
        st.markdown(f"""
            <div style='text-align: center; font-size: 32px; font-weight: bold; color: #2E86AB; margin-top: 20px;'>
                Best Model: {model_name}
            </div>
        """, unsafe_allow_html=True)
    else:
        st.info("Run a model first from the Upload tab.")

with tabs[2]:
    st.subheader("🧠 Feature Importance (SHAP)")
    if 'automl' in st.session_state:
        try:
            raw_model = (st.session_state['automl'].model.estimator
                         if hasattr(st.session_state['automl'].model, 'estimator')
                         else st.session_state['automl'].model)

            if hasattr(raw_model, 'fit') and hasattr(raw_model, 'predict'):
                explainer = shap.Explainer(raw_model, st.session_state['X_train'])
                shap_values = explainer(st.session_state['X_test'])

                fig, ax = plt.subplots()
                shap.summary_plot(shap_values, st.session_state['X_test'], show=False)
                st.pyplot(fig)
            else:
                st.warning("Model is not SHAP-compatible or unfitted.")
        except Exception as e:
            st.warning(f"⚠️ SHAP explanation could not be generated: {e}")
    else:
        st.info("Run a model first from the Upload tab.")
