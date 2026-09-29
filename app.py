import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .sub-title {
        font-size: 18px;
        color: #9ca3af;
        margin-bottom: 25px;
    }

    .section-title {
        font-size: 28px;
        font-weight: 650;
        margin-top: 15px;
        margin-bottom: 15px;
    }

    .info-box {
        padding: 18px;
        border-radius: 12px;
        border: 1px solid rgba(128,128,128,0.25);
        background: rgba(128,128,128,0.06);
    }

    .success-box {
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #22c55e;
        background: rgba(34,197,94,0.08);
    }

    .danger-box {
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #ef4444;
        background: rgba(239,68,68,0.08);
    }

    </style>
    """,
    unsafe_allow_html=True
)

# ============================================================
# LOAD MODEL
# ============================================================

MODEL_PATH = Path(__file__).parent / "customer_churn_model.pkl"


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


try:
    model = load_model()

except Exception as e:
    st.error(f"❌ Could not load the model: {e}")
    st.stop()


# ============================================================
# LOAD MODEL COMPARISON DATA
# ============================================================

CSV_PATH = Path(__file__).parent / "ML_Model_Comparison.csv"


@st.cache_data
def load_comparison_data():

    if CSV_PATH.exists():

        df = pd.read_csv(CSV_PATH)

        return df

    # Backup values from the model comparison
    return pd.DataFrame(
        {
            "Model": [
                "Random Forest",
                "KNN",
                "Naive Bayes",
                "SVM",
                "Decision Tree"
            ],
            "Training Accuracy": [
                100.00,
                92.04,
                75.70,
                93.91,
                100.00
            ],
            "Testing Accuracy": [
                92.50,
                88.98,
                74.64,
                93.02,
                85.96
            ],
            "Precision": [
                92.41,
                86.36,
                64.08,
                93.68,
                79.99
            ],
            "Recall": [
                85.89,
                81.83,
                64.73,
                86.12,
                80.53
            ],
            "F1 Score": [
                89.03,
                84.03,
                64.40,
                89.74,
                80.26
            ]
        }
    )


comparison_df = load_comparison_data()

# ============================================================
# BEST MODEL
# ============================================================

best_row = comparison_df.loc[
    comparison_df["Testing Accuracy"].idxmax()
]

best_model_name = best_row["Model"]
best_accuracy = best_row["Testing Accuracy"]


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("📊 Churn Dashboard")

    st.markdown("---")

    st.markdown(
        """
        ### Project

        **Customer Churn Prediction**

        Machine Learning classification project using multiple ML algorithms.

        ### Models

        - 🌲 Random Forest
        - 🔹 KNN
        - 🧮 Naive Bayes
        - 🎯 SVM
        - 🌳 Decision Tree

        ### Deployment

        🚀 Streamlit Community Cloud
        """
    )

    st.markdown("---")

    st.caption("Built by Anjani Kumar")


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">📊 Customer Churn Prediction Dashboard</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">'
    'Machine Learning dashboard for customer churn analysis and prediction.'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# NAVIGATION TABS
# ============================================================

dashboard_tab, prediction_tab, model_tab, about_tab = st.tabs(
    [
        "📊 Dashboard",
        "🎯 Predict Churn",
        "🤖 Model Performance",
        "ℹ️ About Project"
    ]
)


# ============================================================
# DASHBOARD TAB
# ============================================================

with dashboard_tab:

    st.markdown(
        '<div class="section-title">📈 Project Overview</div>',
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # KPI CARDS
    # --------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "🤖 Models Compared",
            len(comparison_df)
        )

    with col2:

        st.metric(
            "🏆 Best Model",
            best_model_name
        )

    with col3:

        st.metric(
            "🎯 Best Testing Accuracy",
            f"{best_accuracy:.2f}%"
        )

    with col4:

        st.metric(
            "📊 Evaluation Metrics",
            "5"
        )

    st.divider()

    # --------------------------------------------------------
    # ACCURACY GRAPH
    # --------------------------------------------------------

    st.subheader("📈 Training vs Testing Accuracy")

    accuracy_chart = comparison_df[
        [
            "Model",
            "Training Accuracy",
            "Testing Accuracy"
        ]
    ].copy()

    accuracy_chart = accuracy_chart.set_index("Model")

    st.bar_chart(
        accuracy_chart,
        use_container_width=True
    )

    st.caption(
        "Comparison of training and testing accuracy across the five classification models."
    )

    st.divider()

    # --------------------------------------------------------
    # PERFORMANCE METRICS
    # --------------------------------------------------------

    st.subheader("📊 Model Performance Metrics")

    metrics_chart = comparison_df[
        [
            "Model",
            "Precision",
            "Recall",
            "F1 Score"
        ]
    ].copy()

    metrics_chart = metrics_chart.set_index("Model")

    st.bar_chart(
        metrics_chart,
        use_container_width=True
    )

    st.caption(
        "Precision, recall and F1 score for each machine learning model."
    )

    st.divider()

    # --------------------------------------------------------
    # BEST MODEL CARD
    # --------------------------------------------------------

    st.subheader("🏆 Selected Model")

    best_col1, best_col2 = st.columns(2)

    with best_col1:

        st.markdown(
            f"""
            <div class="info-box">

            ### 🎯 {best_model_name}

            **Testing Accuracy:** {best_accuracy:.2f}%

            </div>
            """,
            unsafe_allow_html=True
        )

    with best_col2:

        st.markdown(
            """
            <div class="info-box">

            ### 🔬 Evaluation

            Models were evaluated using:

            - Accuracy
            - Precision
            - Recall
            - F1 Score

            </div>
            """,
            unsafe_allow_html=True
        )

    st.divider()

    # --------------------------------------------------------
    # MODEL TABLE
    # --------------------------------------------------------

    st.subheader("📋 Complete Model Comparison")

    display_df = comparison_df.copy()

    numeric_columns = [
        "Training Accuracy",
        "Testing Accuracy",
        "Precision",
        "Recall",
        "F1 Score"
    ]

    for column in numeric_columns:

        display_df[column] = display_df[column].map(
            lambda x: f"{x:.2f}%"
        )

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# PREDICTION TAB
# ============================================================

with prediction_tab:

    st.markdown(
        '<div class="section-title">🎯 Customer Churn Prediction</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Enter customer information below to estimate the customer's churn risk."
    )

    # --------------------------------------------------------
    # GET PREPROCESSOR INFORMATION
    # --------------------------------------------------------

    try:

        preprocessor = model.named_steps["preprocessor"]

    except Exception:

        st.error(
            "The saved model does not contain the expected preprocessing pipeline."
        )

        st.stop()


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


    # --------------------------------------------------------
    # FEATURE ORDER
    # --------------------------------------------------------

    feature_names = list(model.feature_names_in_)


    # --------------------------------------------------------
    # CUSTOMER FORM
    # --------------------------------------------------------

    with st.form("customer_prediction_form"):

        st.subheader("👤 Customer Information")

        customer_data = {}

        left, right = st.columns(2)

        for index, feature in enumerate(feature_names):

            container = (
                left
                if index % 2 == 0
                else right
            )

            with container:

                label = (
                    feature
                    .replace("_", " ")
                    .title()
                )

                # ------------------------------
                # NUMERICAL FEATURES
                # ------------------------------

                if feature in numeric_features:

                    customer_data[feature] = st.number_input(
                        label,
                        value=0.0,
                        format="%.2f"
                    )

                # ------------------------------
                # CATEGORICAL FEATURES
                # ------------------------------

                elif feature in categorical_features:

                    options = categorical_categories.get(
                        feature,
                        []
                    )

                    if options:

                        customer_data[feature] = st.selectbox(
                            label,
                            options=options
                        )

                    else:

                        customer_data[feature] = st.text_input(
                            label
                        )

                # ------------------------------
                # FALLBACK
                # ------------------------------

                else:

                    customer_data[feature] = st.text_input(
                        label
                    )


        submitted = st.form_submit_button(
            "🔮 Predict Churn",
            type="primary",
            use_container_width=True
        )


    # ========================================================
    # PREDICTION RESULT
    # ========================================================

    if submitted:

        input_df = pd.DataFrame(
            [customer_data],
            columns=feature_names
        )

        try:

            prediction = model.predict(input_df)[0]

            st.divider()

            st.subheader("🔮 Prediction Result")

            result_col1, result_col2 = st.columns(2)

            # ------------------------------------------------
            # PREDICTED CLASS
            # ------------------------------------------------

            with result_col1:

                st.metric(
                    "Predicted Class",
                    str(prediction)
                )

            # ------------------------------------------------
            # CHURN PROBABILITY
            # ------------------------------------------------

            churn_probability = None

            if hasattr(model, "predict_proba"):

                probabilities = model.predict_proba(
                    input_df
                )[0]

                classes = list(model.classes_)

                churn_index = None

                for i, label in enumerate(classes):

                    normalized_label = (
                        str(label)
                        .strip()
                        .lower()
                    )

                    if normalized_label in [
                        "1",
                        "yes",
                        "true",
                        "churn"
                    ]:

                        churn_index = i
                        break


                if churn_index is not None:

                    churn_probability = (
                        probabilities[churn_index]
                    )

                    with result_col2:

                        st.metric(
                            "Churn Probability",
                            f"{churn_probability:.1%}"
                        )

                else:

                    with result_col2:

                        st.info(
                            "Churn probability could not be determined."
                        )


            # ------------------------------------------------
            # RESULT MESSAGE
            # ------------------------------------------------

            normalized_prediction = (
                str(prediction)
                .strip()
                .lower()
            )

            if normalized_prediction in [
                "1",
                "yes",
                "true",
                "churn"
            ]:

                st.markdown(
                    """
                    <div class="danger-box">

                    ## ⚠️ Churn Risk Detected

                    The machine learning model predicts that
                    this customer may churn.

                    </div>
                    """,
                    unsafe_allow_html=True
                )

            else:

                st.markdown(
                    """
                    <div class="success-box">

                    ## ✅ Customer Likely to Stay

                    The machine learning model predicts that
                    this customer may stay.

                    </div>
                    """,
                    unsafe_allow_html=True
                )


            # ------------------------------------------------
            # PROBABILITY BAR
            # ------------------------------------------------

            if churn_probability is not None:

                st.subheader("📊 Churn Risk Level")

                st.progress(
                    float(churn_probability)
                )

                if churn_probability >= 0.70:

                    st.error(
                        f"High churn risk: {churn_probability:.1%}"
                    )

                elif churn_probability >= 0.40:

                    st.warning(
                        f"Medium churn risk: {churn_probability:.1%}"
                    )

                else:

                    st.success(
                        f"Low churn risk: {churn_probability:.1%}"
                    )


            # ------------------------------------------------
            # INPUT DATA
            # ------------------------------------------------

            with st.expander(
                "🔍 View Customer Input"
            ):

                st.dataframe(
                    input_df,
                    use_container_width=True,
                    hide_index=True
                )


        except Exception as e:

            st.error(
                f"❌ Prediction failed: {e}"
            )


# ============================================================
# MODEL PERFORMANCE TAB
# ============================================================

with model_tab:

    st.markdown(
        '<div class="section-title">🤖 Machine Learning Model Performance</div>',
        unsafe_allow_html=True
    )

    st.write(
        "The project compares five classification algorithms."
    )

    # --------------------------------------------------------
    # TESTING ACCURACY
    # --------------------------------------------------------

    st.subheader("🎯 Testing Accuracy")

    testing_accuracy = comparison_df[
        [
            "Model",
            "Testing Accuracy"
        ]
    ].copy()

    testing_accuracy = testing_accuracy.set_index(
        "Model"
    )

    st.bar_chart(
        testing_accuracy,
        use_container_width=True
    )

    # --------------------------------------------------------
    # DETAILED METRICS
    # --------------------------------------------------------

    st.subheader("📊 Detailed Evaluation")

    st.dataframe(
        comparison_df,
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------------------------------
    # BEST MODEL DETAILS
    # --------------------------------------------------------

    st.subheader("🏆 Best Performing Model")

    st.write(
        f"**Model:** {best_model_name}"
    )

    st.write(
        f"**Testing Accuracy:** {best_accuracy:.2f}%"
    )

    st.write(
        f"**Precision:** {best_row['Precision']:.2f}%"
    )

    st.write(
        f"**Recall:** {best_row['Recall']:.2f}%"
    )

    st.write(
        f"**F1 Score:** {best_row['F1 Score']:.2f}%"
    )


# ============================================================
# ABOUT TAB
# ============================================================

with about_tab:

    st.markdown(
        '<div class="section-title">ℹ️ About This Project</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        ## Customer Churn Prediction

        This project uses machine learning classification algorithms
        to predict whether a customer is likely to churn.

        ### 🛠️ Technologies Used

        - Python
        - Pandas
        - NumPy
        - Scikit-learn
        - Joblib
        - Streamlit

        ### 🤖 Machine Learning Models

        - Random Forest
        - K-Nearest Neighbors (KNN)
        - Gaussian Naive Bayes
        - Support Vector Machine (SVM)
        - Decision Tree

        ### 📊 Evaluation Metrics

        - Training Accuracy
        - Testing Accuracy
        - Precision
        - Recall
        - F1 Score

        ### 🔄 ML Workflow

        **Data → Preprocessing → Train/Test Split → Encoding →
        Scaling → Model Training → Model Comparison →
        Best Model → Streamlit Deployment**

        ### 🚀 Deployment

        The application is deployed using Streamlit Community Cloud.

        ---
        
        **Built by Anjani Kumar**
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "📊 Customer Churn Prediction | Machine Learning Project | Built by Anjani Kumar"
) 
