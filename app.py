import streamlit as st
import joblib

# Load the trained model
model = joblib.load("loan_classifier.joblib")

def predict_loan_status(
    int_rate,
    installment,
    log_annual_inc,
    dti,
    fico,
    revol_bal,
    revol_util,
    inq_last_6mths,
    delinq_2yrs,
    pub_rec,
    installment_to_income_ratio,
    credit_history,
):
    input_dict = {
        "int.rate": int_rate,
        "installment": installment,
        "log.annual.inc": log_annual_inc,
        "dti": dti,
        "fico": fico,
        "revol.bal": revol_bal,
        "revol.util": revol_util,
        "inq.last.6mths": inq_last_6mths,
        "delinq.2yrs": delinq_2yrs,
        "pub.rec": pub_rec,
        "installment_to_income_ratio": installment_to_income_ratio,
        "credit_history": credit_history,
    }
    # Convert the dictionary to a 2D array
    input_array = [list(input_dict.values())]
    prediction = model.predict(input_array)[0]

    if prediction == 0:
        return "Loan fully paid"
    else:
        return "Loan not fully paid"

st.title("Loan Approval Classifier")
st.write("Enter the details of the loan applicant to check if the loan is approved or not.")

int_rate = st.slider("Interest Rate", 0.06, 0.23, step=0.01)
installment = st.slider("Installment", 100, 950, step=10)
log_annual_inc = st.slider("Log Annual Income", 7.0, 15.0, step=0.1)  # Use floats here
dti = st.slider("DTI Ratio", 0, 40, step=1)
fico = st.slider("FICO Score", 600, 850, step=1)
revol_bal = st.slider("Revolving Balance", 0, 120000, step=1000)
revol_util = st.slider("Revolving Utilization", 0, 120, step=1)
inq_last_6mths = st.slider("Inquiries in Last 6 Months", 0, 10, step=1)
delinq_2yrs = st.slider("Delinquencies in Last 2 Years", 0, 20, step=1)
pub_rec = st.slider("Public Records", 0, 10, step=1)
installment_to_income_ratio = st.slider("Installment to Income Ratio", 0.0, 5.0, step=0.1)  # Use floats here
credit_history = st.slider("Credit History", 0.0, 1.0, step=0.01)

if st.button("Predict"):
    result = predict_loan_status(
        int_rate,
        installment,
        log_annual_inc,
        dti,
        fico,
        revol_bal,
        revol_util,
        inq_last_6mths,
        delinq_2yrs,
        pub_rec,
        installment_to_income_ratio,
        credit_history,
    )
    st.write(result)
