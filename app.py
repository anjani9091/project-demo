import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# ----------------------------------
# Page settings
# ----------------------------------
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# ----------------------------------
# Load trained model
# ----------------------------------
MODEL_PATH = Path(__file__).parent / "customer_churn_model.pkl"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

try:
    model = load_model()
except Exception as e:
    st.error(f"Could not load the model: {e}")
    st.stop()

# ----------------------------------
# App heading
# ----------------------------------
st.title("📊 Customer Churn Prediction")
st.write(
    "Enter customer details to estimate churn risk "
    "using your trained machine learning model."
)

st.divider()

# Get the preprocessing step from the saved pipeline
preprocessor = model.named_steps["preprocessor"]

numeric_features = []
categorical_features = []
categorical_categories = {}

for name, transformer, columns in preprocessor.transformers_:
    if name == "num":
        numeric_features = list(columns)

    elif name == "cat":
        categorical_features = list(columns)

        encoder = transformer
        for column, categories in zip(
            categorical_features,
            encoder.categories_
        ):
            categorical_categories[column] = list(categories)

# Keep the same column order used during training
feature_names = list(model.feature_names_in_)

# ----------------------------------
# Customer input form
# ----------------------------------
st.subheader("Customer Information")

with st.form("customer_form"):
    customer_data = {}

    left, right = st.columns(2)

    for index, feature in enumerate(feature_names):
        container = left if index % 2 == 0 else right

        with container:
            if feature in numeric_features:
                customer_data[feature] = st.number_input(
                    feature.replace("_", " ").title(),
                    value=0.0,
                    format="%.2f"
                )

            elif feature in categorical_features:
                options = categorical_categories.get(feature, [])

                if options:
                    customer_data[feature] = st.selectbox(
                        feature.replace("_", " ").title(),
                        options=options
                    )
                else:
                    customer_data[feature] = st.text_input(
                        feature.replace("_", " ").title()
                    )

    submitted = st.form_submit_button(
        "Predict Churn",
        type="primary",
        use_container_width=True
    )

# ----------------------------------
# Prediction
# ----------------------------------
if submitted:
    input_df = pd.DataFrame(
        [customer_data],
        columns=feature_names
    )

    try:
        prediction = model.predict(input_df)[0]

        st.divider()
        st.subheader("Prediction Result")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Predicted Class", str(prediction))

        with col2:
            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(input_df)[0]
                classes = list(model.classes_)

                # target_churn = 1 represents churn
                churn_index = next(
                    (
                        i for i, label in enumerate(classes)
                        if str(label).strip().lower()
                        in ["1", "yes", "true", "churn"]
                    ),
                    None
                )

                if churn_index is not None:
                    churn_probability = probabilities[churn_index]
                    st.metric(
                        "Churn Probability",
                        f"{churn_probability:.1%}"
                    )
                    st.progress(float(churn_probability))
                else:
                    st.info(
                        "Churn probability could not be identified "
                        "from the model's class labels."
                    )

        if str(prediction).strip().lower() in [
            "1", "yes", "true", "churn"
        ]:
            st.error("⚠️ The model predicts that this customer may churn.")
        else:
            st.success("✅ The model predicts that this customer may stay.")

    except Exception as e:
        st.error(f"Prediction failed: {e}")

st.divider()
st.caption("Machine Learning Project | Built by Anjani Kumar")
