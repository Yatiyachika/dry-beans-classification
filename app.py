import streamlit as st
import numpy as np
import pandas as pd
import joblib

st.set_page_config(
    page_title="Dry Bean Classification",
    page_icon="🫘",
    layout="wide"
)

# Actual class list in exact LabelEncoder alphabetical order
CLASSES = ['BARBUNYA', 'BOMBAY', 'CALI', 'DERMASON', 'HOROZ', 'SEKER', 'SIRA']

@st.cache_resource
def load_artifacts():
    model = joblib.load('dry_bean_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

try:
    model, scaler = load_artifacts()
except Exception as e:
    st.error("⚠️ Model or Scaler load nahi hua! Check karo `.pkl` files same folder me hain.")
    st.stop()

st.title("🫘 Dry Bean Species Classification App")
st.write("Provide the morphological features below to predict the species/class of the dry bean.")

st.sidebar.header("Input Features")

def user_input_features():
    area = st.sidebar.number_input("Area", min_value=10000.0, max_value=300000.0, value=50000.0)
    perimeter = st.sidebar.number_input("Perimeter", min_value=100.0, max_value=3000.0, value=850.0)
    major_axis_length = st.sidebar.number_input("Major Axis Length", min_value=100.0, max_value=1000.0, value=300.0)
    minor_axis_length = st.sidebar.number_input("Minor Axis Length", min_value=50.0, max_value=600.0, value=200.0)
    aspect_ratio = st.sidebar.number_input("Aspect Ratio", min_value=1.0, max_value=3.0, value=1.5)
    eccentricity = st.sidebar.number_input("Eccentricity", min_value=0.0, max_value=1.0, value=0.75)
    convex_area = st.sidebar.number_input("Convex Area", min_value=10000.0, max_value=300000.0, value=51000.0)
    equiv_diameter = st.sidebar.number_input("Equiv Diameter", min_value=100.0, max_value=1000.0, value=250.0)
    extent = st.sidebar.number_input("Extent", min_value=0.0, max_value=1.0, value=0.7)
    solidity = st.sidebar.number_input("Solidity", min_value=0.0, max_value=1.0, value=0.98)
    roundness = st.sidebar.number_input("Roundness", min_value=0.0, max_value=1.0, value=0.85)
    compactness = st.sidebar.number_input("Compactness", min_value=0.0, max_value=1.0, value=0.8)
    shape_factor_1 = st.sidebar.number_input("Shape Factor 1", min_value=0.0, max_value=0.1, value=0.006, format="%.6f")
    shape_factor_2 = st.sidebar.number_input("Shape Factor 2", min_value=0.0, max_value=0.1, value=0.001, format="%.6f")
    shape_factor_3 = st.sidebar.number_input("Shape Factor 3", min_value=0.0, max_value=1.0, value=0.6, format="%.6f")
    shape_factor_4 = st.sidebar.number_input("Shape Factor 4", min_value=0.0, max_value=1.0, value=0.99, format="%.6f")

    data = {
        'Area': area, 'Perimeter': perimeter, 'MajorAxisLength': major_axis_length,
        'MinorAxisLength': minor_axis_length, 'AspectRation': aspect_ratio, 'Eccentricity': eccentricity,
        'ConvexArea': convex_area, 'EquivDiameter': equiv_diameter, 'Extent': extent,
        'Solidity': solidity, 'roundness': roundness, 'Compactness': compactness,
        'ShapeFactor1': shape_factor_1, 'ShapeFactor2': shape_factor_2,
        'ShapeFactor3': shape_factor_3, 'ShapeFactor4': shape_factor_4
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

st.subheader("Selected Input Parameters")
st.dataframe(input_df)

if st.button("Predict Species"):
    skewed_cols = ['Area', 'ConvexArea', 'MajorAxisLength', 'MinorAxisLength', 'Perimeter', 'EquivDiameter']
    
    input_transformed = input_df.copy()
    for col in skewed_cols:
        if col in input_transformed.columns:
            input_transformed[col] = np.log1p(input_transformed[col])
            
    scaled_input = scaler.transform(input_transformed)
    
    # Raw numeric prediction
    pred_raw = model.predict(scaled_input)[0]
    
    # Map raw number to actual String name
    if isinstance(pred_raw, (int, np.integer)):
        predicted_class_name = CLASSES[int(pred_raw)]
    else:
        predicted_class_name = str(pred_raw)
        
    st.success(f"✅ **Predicted Bean Class:** `{predicted_class_name}`")

    # Probabilities Chart with Exact String Labels
    st.subheader("Prediction Confidence / Probabilities")
    prediction_proba = model.predict_proba(scaled_input)
    proba_df = pd.DataFrame(prediction_proba, columns=CLASSES)
    st.bar_chart(proba_df.T)