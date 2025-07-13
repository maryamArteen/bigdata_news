import streamlit as st
import joblib
import os

st.title("Fake News Detection")

# Debug information
st.write("**Debug Info:**")
st.write(f"Current working directory: {os.getcwd()}")
st.write(f"Files in current directory: {os.listdir('.')}")

# Check if Models folder exists
if os.path.exists('Models'):
    st.write(f"✅ Models folder found")
    st.write(f"Files in Models folder: {os.listdir('Models')}")
    
    # Check if the specific file exists
    model_path = 'Models/fake_news_rf_pipeline.pkl'
    if os.path.exists(model_path):
        st.write(f"✅ Model file found at: {model_path}")
        file_size = os.path.getsize(model_path)
        st.write(f"File size: {file_size} bytes ({file_size/1024/1024:.1f} MB)")
    else:
        st.write(f"❌ Model file NOT found at: {model_path}")
else:
    st.write("❌ Models folder NOT found")

# Try to load model with better error handling
st.write("**Attempting to load model:**")

try:
    pipeline = joblib.load('Models/fake_news_rf_pipeline.pkl')
    st.success("✅ Model loaded successfully!")
    
    # Test prediction
    test_headline = "This is a test headline"
    prediction = pipeline.predict([test_headline])[0]
    st.write(f"Test prediction works: {prediction}")
    
except FileNotFoundError as e:
    st.error(f"❌ File not found: {e}")
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.write(f"Error type: {type(e)}")