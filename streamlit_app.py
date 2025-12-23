import streamlit as st
import pandas as pd
import joblib
from io import StringIO

# Load model and scaler
model = joblib.load('loaneligibilitypredictor.pkl')
scaler = joblib.load('scaler.pkl')

# App UI Configuration
st.set_page_config(
    page_title="Loan Eligibility Predictor",
    page_icon="🏦",
    layout="centered"
)

# Custom CSS for better radio/select controls
st.markdown("""
    <style>
        /* Improved radio buttons */
        .stRadio > div {
            flex-direction: row;
            gap: 15px;
        }
        .stRadio [role="radiogroup"] {
            align-items: center;
            gap: 15px;
        }

        /* Better select boxes */
        .stSelectbox [data-baseweb="select"] {
            min-width: 200px;
        }

        /* Number input improvements */
        .stNumberInput input {
            min-width: 200px !important;
        }
    </style>
""", unsafe_allow_html=True)

# Header Section
st.title("🏦 Loan Eligibility Predictor")
st.markdown("Complete the application form to check loan eligibility")

# Input Form
with st.form("loan_form"):
    st.subheader("Personal Information")

    col1, col2 = st.columns(2)
    with col1:
        # Radio buttons for binary choices
        gender = st.radio("Gender", ["Male", "Female"], horizontal=True)
        married = st.radio("Married", ["Yes", "No"], horizontal=True)

    with col2:
        # Better number input for dependents
        dependents = st.slider("Number of Dependents", 0, 4, 0)
        education = st.radio("Education", ["Graduate", "Not Graduate"], horizontal=True)

    st.divider()
    st.subheader("Employment Details")

    self_employed = st.radio("Self Employed", ["Yes", "No"], horizontal=True)
    property_area = st.selectbox("Property Area",
                                 ["Urban", "Semiurban", "Rural"],
                                 index=0)

    st.divider()
    st.subheader("Financial Information")

    col1, col2 = st.columns(2)
    with col1:
        applicant_income = st.number_input("Applicant Income ($)",
                                           min_value=645.0,
                                           step=1000.0,
                                           value=5000.0)
        coapplicant_income = st.number_input("Co-applicant Income ($)",
                                             min_value=0.0,
                                             step=500.0,
                                             value=2000.0)
    with col2:
        loan_amount = st.number_input("Loan Amount ($)",
                                      min_value=100.0,
                                      step=1000.0,
                                      value=150000.0)
        loan_amount_term = st.slider("Loan Term (months)",
                                     min_value=36,
                                     max_value=480,
                                     value=180,
                                     step=12)

    credit_history = st.radio("Credit History",
                              options=[1.0, 0.0],
                              format_func=lambda x: "Good Credit" if x == 1.0 else "Bad Credit",
                              horizontal=True)

    submitted = st.form_submit_button("Check Eligibility", type="primary")

# Process form submission
if submitted:
    # Create input dictionary
    input_dict = {
        'dependents': float(dependents),
        'applicant_income': applicant_income,
        'coapplicant_income': coapplicant_income,
        'loan_amount': loan_amount,
        'loan_amount_term': loan_amount_term,
        'credit_history': credit_history,
        'gender': 1 if gender == 'Male' else 0,
        'married': 1 if married == 'Yes' else 0,
        'education': 1 if education == 'Graduate' else 0,
        'self_employed': 1 if self_employed == 'Yes' else 0,
        'property_area': {'Rural': 0, 'Urban': 1, 'Semiurban': 2}[property_area]
    }

    # Create DataFrame
    input_df = pd.DataFrame([input_dict])
    original_df = input_df.copy()

    # Scale numerical features
    numerical_cols = ['dependents', 'applicant_income', 'coapplicant_income',
                      'loan_amount', 'loan_amount_term', 'credit_history']
    input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])

    # Make prediction
    prediction = model.predict(input_df)[0]
    result = "Approved" if prediction == 1 else "Not Approved"

    # Display Results
    st.subheader("Loan Decision")
    if prediction == 1:
        st.success("✅ Loan Approved!")
    else:
        st.error("❌ Loan Not Approved")

    # Show details in expandable section
    with st.expander("View Application Details"):
        st.dataframe(original_df.style.format({
            'applicant_income': '${:,.2f}',
            'coapplicant_income': '${:,.2f}',
            'loan_amount': '${:,.2f}'
        }))

    # Download Section
    original_df['prediction'] = result
    csv = original_df.to_csv(index=False)

    st.download_button(
        label="Download Application Details",
        data=csv,
        file_name="loan_application_result.csv",
        mime="text/csv"
    )