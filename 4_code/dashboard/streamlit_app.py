import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler

st.title("PandemicGuard: AI Pandemic Early Detection")

# Load model and preprocessors (adjust paths as needed)
@st.cache(allow_output_mutation=True)
def load_artifacts():
    model = joblib.load('./models/pandemicguard_model.pkl')
    encoders = joblib.load('./models/label_encoders.pkl')
    scaler = joblib.load('./models/scaler.pkl')
    return model, encoders, scaler

model, encoders, scaler = load_artifacts()

st.write("Upload new data CSV to predict potential pandemic signals:")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Raw Input Data:")
    st.dataframe(df.head())

    # Preprocess
    for col, le in encoders.items():
        if col in df.columns:
            try:
                df[col] = le.transform(df[col].fillna('Unknown'))
            except Exception as e:
                st.error(f"Error encoding column '{col}': {e}")

    num_cols = df.select_dtypes(include=['float64', 'int64']).columns
    try:
        df[num_cols] = scaler.transform(df[num_cols])
    except Exception as e:
        st.error(f"Error scaling numerical features: {e}")

    # Predict
    try:
        predictions = model.predict(df)
        df['Prediction'] = predictions
        st.write("Predictions:")
        st.dataframe(df[['Prediction']].head())
    except Exception as e:
        st.error(f"Prediction error: {e}")

else:
    st.info("Please upload a CSV file to get predictions.")
