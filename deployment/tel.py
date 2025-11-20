import streamlit as st
import pandas as pd
import joblib

# Load model and scaler (ensure you ran train_and_save.py first)
model = joblib.load(r"\gradient_boosting_model.joblib")
scaler = joblib.load(r"\scaler.joblib")

# Automatically get the exact feature order the model expects
feature_order = list(model.feature_names_in_)

# Numeric columns used during training (no charge columns except total_charge)
numeric_cols = [
    'account_length', 'voice_mail_messages', 'day_mins', 'evening_mins',
    'night_mins', 'international_mins', 'customer_service_calls',
    'day_calls', 'evening_calls', 'night_calls', 'international_calls',
    'total_charge'
]

st.set_page_config(page_title="Customer Churn Prediction", page_icon="📞")
st.title("📊 Customer Churn Prediction App")
st.write("Predict whether a customer will stay or leave based on their telecom usage data.")

# Sidebar categorical inputs
st.sidebar.header("📋 Customer Plan Details")
voice_mail_plan = st.sidebar.selectbox("Has Voice Mail Plan?", [0, 1],
                                       format_func=lambda x: "Yes" if x == 1 else "No")
international_plan = st.sidebar.selectbox("Has International Plan?", [0, 1],
                                          format_func=lambda x: "Yes" if x == 1 else "No")

# Numeric inputs
st.header("📞 Customer Usage Details")
account_length = st.number_input("Account Length (days)", min_value=0, max_value=500, value=100)
voice_mail_messages = st.number_input("Voice Mail Messages", min_value=0, max_value=100, value=20)
day_mins = st.number_input("Daytime Minutes", min_value=0.0, max_value=500.0, value=250.0)
evening_mins = st.number_input("Evening Minutes", min_value=0.0, max_value=500.0, value=200.0)
night_mins = st.number_input("Night Minutes", min_value=0.0, max_value=500.0, value=150.0)
international_mins = st.number_input("International Minutes", min_value=0.0, max_value=100.0, value=10.0)
customer_service_calls = st.number_input("Customer Service Calls", min_value=0, max_value=20, value=2)
day_calls = st.number_input("Day Calls", min_value=0, max_value=200, value=110)
evening_calls = st.number_input("Evening Calls", min_value=0, max_value=200, value=100)
night_calls = st.number_input("Night Calls", min_value=0, max_value=200, value=90)
international_calls = st.number_input("International Calls", min_value=0, max_value=20, value=3)
total_charge = st.number_input("Total Monthly Charge", min_value=0.0, max_value=300.0, value=69.2)

# Build input dataframe
input_data = pd.DataFrame({
    'account_length': [account_length],
    'voice_mail_plan': [voice_mail_plan],
    'voice_mail_messages': [voice_mail_messages],
    'day_mins': [day_mins],
    'evening_mins': [evening_mins],
    'night_mins': [night_mins],
    'international_mins': [international_mins],
    'customer_service_calls': [customer_service_calls],
    'international_plan': [international_plan],
    'day_calls': [day_calls],
    'evening_calls': [evening_calls],
    'night_calls': [night_calls],
    'international_calls': [international_calls],
    'total_charge': [total_charge]
})

# Reorder columns exactly as model expects
input_data = input_data[feature_order]

# Scale only numeric columns (must match scaler used during training)
input_scaled = input_data.copy()
input_scaled[numeric_cols] = scaler.transform(input_scaled[numeric_cols])

# Predict churn
if st.button("🔍 Predict Churn"):
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[:, 1][0]

    if prediction == 0:
        st.success(f"✅ Prediction: Customer Stayed (Probability of Leaving: {probability:.2%})")
    else:
        st.error(f"⚠️ Prediction: Customer Left (Probability of Leaving: {probability:.2%})")

st.markdown("---")