import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os
import time

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# LOAD MODELS & OBJECTS
# =========================
@st.cache_resource
def load_models():
    try:
        model = joblib.load("heart_model.pkl")
        encoders = joblib.load("encoders.pkl")
        scaler = joblib.load("scaler.pkl")
        return model, encoders, scaler
    except FileNotFoundError as e:
        st.error(f"❌ Error loading model files: {e}")
        st.stop()

model, encoders, scaler = load_models()

# =========================
# STUNNING ANIMATIONS & STYLING
# =========================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    /* MODERN ULTRA-CLEAN DESIGN */
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        letter-spacing: -0.01em;
    }
    
    /* Pure black minimalist background */
    .main {
        background-color: #000000;
    }
    
    /* Modern sidebar with refined gradient */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a0a0f 0%, #1a1a2e 50%, #16213e 100%);
        box-shadow: 4px 0 24px rgba(0, 0, 0, 0.5);
        animation: slideIn 0.6s cubic-bezier(0.16, 1, 0.3, 1);
    }
    
    @keyframes slideIn {
        from {
            transform: translateX(-100%);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
    
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3, 
    [data-testid="stSidebar"] label {
        color: #ffffff !important;
        font-weight: 500 !important;
    }
    
    /* Ultra-modern glassmorphic cards */
    .metric-card {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.1), rgba(255, 255, 255, 0.05));
        backdrop-filter: blur(20px) saturate(180%);
        -webkit-backdrop-filter: blur(20px) saturate(180%);
        border: 1px solid rgba(255, 255, 255, 0.18);
        border-radius: 24px;
        padding: 48px 36px;
        margin: 24px 0;
        text-align: center;
        box-shadow: 
            0 8px 32px rgba(0, 0, 0, 0.3),
            inset 0 1px 0 rgba(255, 255, 255, 0.1);
        transition: all 0.5s cubic-bezier(0.16, 1, 0.3, 1);
        animation: fadeUp 0.7s ease;
        position: relative;
        overflow: hidden;
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.1), transparent);
        transition: left 0.7s;
    }
    
    .metric-card:hover::before {
        left: 100%;
    }
    
    @keyframes fadeUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .metric-card:hover {
        transform: translateY(-8px);
        box-shadow: 
            0 16px 48px rgba(255, 255, 255, 0.15),
            inset 0 1px 0 rgba(255, 255, 255, 0.2);
        border-color: rgba(255, 255, 255, 0.3);
    }
    
    /* Risk card variants with modern glow */
    .success-card {
        border-left: 3px solid #10b981;
        box-shadow: 
            0 8px 32px rgba(16, 185, 129, 0.15),
            inset 0 1px 0 rgba(255, 255, 255, 0.1);
    }
    
    .success-card:hover {
        box-shadow: 
            0 16px 48px rgba(16, 185, 129, 0.3),
            inset 0 1px 0 rgba(255, 255, 255, 0.2);
    }
    
    .warning-card {
        border-left: 3px solid #f59e0b;
        box-shadow: 
            0 8px 32px rgba(245, 158, 11, 0.15),
            inset 0 1px 0 rgba(255, 255, 255, 0.1);
    }
    
    .warning-card:hover {
        box-shadow: 
            0 16px 48px rgba(245, 158, 11, 0.3),
            inset 0 1px 0 rgba(255, 255, 255, 0.2);
    }
    
    .danger-card {
        border-left: 3px solid #ef4444;
        box-shadow: 
            0 8px 32px rgba(239, 68, 68, 0.15),
            inset 0 1px 0 rgba(255, 255, 255, 0.1);
    }
    
    .danger-card:hover {
        box-shadow: 
            0 16px 48px rgba(239, 68, 68, 0.3),
            inset 0 1px 0 rgba(255, 255, 255, 0.2);
    }
    
    /* Modern button with refined interaction */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 16px;
        padding: 18px 40px;
        font-size: 16px;
        font-weight: 600;
        width: 100%;
        transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1);
        box-shadow: 0 4px 16px rgba(102, 126, 234, 0.3);
        position: relative;
        overflow: hidden;
    }
    
    .stButton>button::before {
        content: '';
        position: absolute;
        top: 50%;
        left: 50%;
        width: 0;
        height: 0;
        border-radius: 50%;
        background: rgba(255, 255, 255, 0.2);
        transform: translate(-50%, -50%);
        transition: width 0.6s, height 0.6s;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(102, 126, 234, 0.4);
    }
    
    .stButton>button:hover::before {
        width: 300px;
        height: 300px;
    }
    
    .stButton>button:active {
        transform: translateY(0);
    }
    
    /* Premium gradient text */
    h1 {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 800;
        font-size: 3.5em;
        text-align: center;
        letter-spacing: -0.02em;
        margin-bottom: 16px;
        animation: shimmer 3s ease-in-out infinite;
    }
    
    @keyframes shimmer {
        0%, 100% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
    }
    
    h2 {
        color: rgba(255, 255, 255, 0.9);
        font-weight: 500;
        font-size: 1.5em;
        letter-spacing: -0.01em;
    }
    
    h3 {
        font-weight: 600;
        letter-spacing: -0.01em;
    }
    
    /* Subtle ambient particles */
    .particle {
        position: fixed;
        width: 6px;
        height: 6px;
        background: radial-gradient(circle, rgba(102, 126, 234, 0.6), transparent);
        border-radius: 50%;
        animation: drift 20s infinite ease-in-out;
        pointer-events: none;
        filter: blur(1px);
    }
    
    @keyframes drift {
        0%, 100% { 
            transform: translateY(0) translateX(0);
            opacity: 0;
        }
        10% { opacity: 0.6; }
        90% { opacity: 0.4; }
        100% { 
            transform: translateY(-100vh) translateX(50px);
            opacity: 0;
        }
    }
    
    /* Streamlit elements refinement */
    .stAlert {
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
    }
    
    /* Modern progress bar */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #667eea, #764ba2);
        border-radius: 8px;
        box-shadow: 0 2px 12px rgba(102, 126, 234, 0.4);
    }
    
    /* Enhanced slider visibility for min/max values */
    [data-testid="stSlider"] {
        padding: 20px 0;
    }
    
    [data-testid="stSlider"] label {
        font-weight: 700 !important;
        font-size: 1.1em !important;
        color: #ffffff !important;
    }
    
    /* Style slider value containers */
    [data-testid="stSlider"] > div {
        color: #ffffff !important;
    }
    
    /* Style the min/max text below slider */
    [data-testid="stSlider"] span {
        color: #ffffff !important;
        font-weight: 700 !important;
        font-size: 1.05em !important;
       
        padding: 6px 12px !important;
        border-radius: 8px !important;
        display: inline-block !important;
        margin: 8px 4px 0 4px !important;
    }

</style>

<!-- Subtle floating particles -->
<div class="particle" style="left: 15%; animation-delay: 0s;"></div>
<div class="particle" style="left: 35%; animation-delay: 3s;"></div>
<div class="particle" style="left: 55%; animation-delay: 6s;"></div>
<div class="particle" style="left: 75%; animation-delay: 2s;"></div>
<div class="particle" style="left: 85%; animation-delay: 5s;"></div>
""", unsafe_allow_html=True)

# =========================
# SIDEBAR - INPUT SECTION
# =========================
st.sidebar.markdown('''
<h1>
    <svg style="width: 1.2em; height: 1.2em; vertical-align: middle; margin-right: 10px;" viewBox="0 0 24 24" fill="#ff4444">
        <path d="M12 21.35l-1.45-1.32C5.4 15.36 2 12.28 2 8.5 2 5.42 4.42 3 7.5 3c1.74 0 3.41.81 4.5 2.09C13.09 3.81 14.76 3 16.5 3 19.58 3 22 5.42 22 8.5c0 3.78-3.4 6.86-8.55 11.54L12 21.35z"/>
    </svg>
    Patient Information
</h1>
''', unsafe_allow_html=True)
st.sidebar.markdown("---")

username = st.sidebar.text_input("👤 Patient Name (Optional)", placeholder="Enter name")

st.sidebar.markdown("### 📋 Demographics")
age = st.sidebar.slider("Age", 20, 100, 50, help="Patient's age in years")
sex = st.sidebar.selectbox("Sex", ["Male", "Female"])

st.sidebar.markdown("---")
st.sidebar.markdown("### 💓 Cardiac Symptoms")

cp = st.sidebar.selectbox(
    "Chest Pain Type",
    ["typical angina", "atypical angina", "non-anginal", "asymptomatic"],
    help="Type of chest pain experienced"
)

exang = st.sidebar.selectbox(
    "Exercise Induced Angina",
    ["False", "True"],
    help="Does exercise cause chest pain?"
)

oldpeak = st.sidebar.slider(
    "ST Depression",
    0.0, 6.0, 1.0, 0.1,
    help="ST depression induced by exercise (ECG reading)"
)

slope = st.sidebar.selectbox(
    "ST Slope",
    ["upsloping", "flat", "downsloping"],
    help="Slope of peak exercise ST segment"
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🩺 Vital Signs & Tests")

trestbps = st.sidebar.slider(
    "Resting Blood Pressure (mm Hg)",
    90, 200, 120,
    help="Blood pressure at rest"
)

chol = st.sidebar.slider(
    "Cholesterol (mg/dl)",
    100, 600, 200,
    help="Serum cholesterol level"
)

thalch = st.sidebar.slider(
    "Maximum Heart Rate",
    60, 220, 150,
    help="Maximum heart rate achieved during exercise"
)

fbs = st.sidebar.selectbox(
    "Fasting Blood Sugar > 120 mg/dl",
    ["False", "True"],
    help="Is fasting blood sugar greater than 120?"
)

restecg = st.sidebar.selectbox(
    "Resting ECG Result",
    ["normal", "st-t abnormality", "lv hypertrophy"],
    help="Resting electrocardiographic results"
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🔬 Advanced Tests")

ca = st.sidebar.slider(
    "Major Vessels (0-3)",
    0, 3, 0,
    help="Number of major vessels colored by fluoroscopy"
)

thal = st.sidebar.selectbox(
    "Thalassemia",
    ["normal", "fixed defect", "reversable defect"],
    help="Thalassemia test result"
)

st.sidebar.markdown("---")
predict_button = st.sidebar.button("🔍 Predict Heart Disease Risk")

# =========================
# MAIN AREA - RESULTS
# =========================
st.markdown('''
<h1>
    <svg style="width: 1em; height: 1em; vertical-align: middle; margin-right: 15px;" viewBox="0 0 24 24" fill="#ff4444">
        <path d="M12 21.35l-1.45-1.32C5.4 15.36 2 12.28 2 8.5 2 5.42 4.42 3 7.5 3c1.74 0 3.41.81 4.5 2.09C13.09 3.81 14.76 3 16.5 3 19.58 3 22 5.42 22 8.5c0 3.78-3.4 6.86-8.55 11.54L12 21.35z"/>
    </svg>
    Heart Disease Prediction System
</h1>
''', unsafe_allow_html=True)
st.markdown('<h2 style="text-align: center; ">AI-Powered Cardiovascular Risk Assessment</h2>', unsafe_allow_html=True)
st.markdown("---")

if not predict_button:
    # Welcome screen with animations
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card" style="animation-delay: 0.1s;">
            <h2 style="color: #667eea; font-size: 3em;">📊</h2>
            <h3 style="color: #2a5298;">Advanced AI Model</h3>
            <p>Binary Random Forest Classifier</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card" style="animation-delay: 0.3s;">
            <h2 style="color: #764ba2; font-size: 3em;">🎯</h2>
            <h3 style="color: #2a5298;">High Accuracy</h3>
            <p>Trained on validated medical data</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card" style="animation-delay: 0.5s;">
            <h2 style="color: #00c851; font-size: 3em;">✅</h2>
            <h3 style="color: #2a5298;">Quick Results</h3>
            <p>Instant risk assessment</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="text-align: center; margin-top: 40px;">
        <p style="font-size: 1.3em;  text-shadow: 0 2px 10px rgba(0,0,0,0.5);">
            👈 Enter patient information in the sidebar and click <strong>Predict</strong> to get started.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
else:
    # Show loading animation
    with st.spinner('🔮 Analyzing data...'):
        time.sleep(1.5)  # Dramatic pause
    
    # PREDICTION LOGIC
    try:
        # Create input DataFrame
        input_df = pd.DataFrame({
            "age": [age],
            "sex": [sex],
            "cp": [cp],
            "trestbps": [trestbps],
            "chol": [chol],
            "fbs": [fbs],
            "restecg": [restecg],
            "thalch": [thalch],
            "exang": [exang],
            "oldpeak": [oldpeak],
            "slope": [slope],
            "ca": [ca],
            "thal": [thal]
        })

        # Apply Label Encoding
        for col in encoders:
            if col in input_df.columns:
                input_df[col] = encoders[col].transform(input_df[col])

        # Apply Scaling
        input_scaled = scaler.transform(input_df)

        # Predict
        prediction = model.predict(input_scaled)[0]
        proba = model.predict_proba(input_scaled)[0]

        # Probability of Heart Disease
        heart_prob = proba[1]

        # Define Risk Levels
        if heart_prob < 0.3:
            risk_label = "Low Risk"
            risk_icon = "✅"
            card_class = "success-card"
            risk_color = "#10b981"  # Green
            recommendation = "Your heart health indicators look excellent! Continue maintaining a healthy lifestyle with regular exercise and a balanced diet."
            emoji_celebration = "🎉"
        elif heart_prob < 0.6:
            risk_label = "Moderate Risk"
            risk_icon = "⚠️"
            card_class = "warning-card"
            risk_color = "#f59e0b"  # Orange
            recommendation = "Some risk factors detected. Consider consulting with a healthcare provider for preventive measures and lifestyle modifications."
            emoji_celebration = "⚡"
        else:
            risk_label = "High Risk"
            risk_icon = "🚨"
            card_class = "danger-card"
            risk_color = "#ef4444"  # Red
            recommendation = "Significant risk factors detected. Please consult a cardiologist as soon as possible for a comprehensive evaluation."
            emoji_celebration = "🆘"

        # Display Results with dramatic reveal
        st.markdown(f'<h2 style="text-align: center; color: white; text-shadow: 0 0 20px rgba(255,255,255,0.5);">{emoji_celebration} Prediction Results {emoji_celebration}</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown(f"""
            <div class="metric-card {card_class}">
                <h1 style="font-size: 4em; margin: 0; background: none; -webkit-text-fill-color: unset; color: unset;">{risk_icon}</h1>
                <h2 style="color: {risk_color}; margin: 20px 0; font-size: 2em;">{risk_label}</h2>
                <p style="font-size: 1.3em; color: #999; font-weight: 600;">Risk Assessment</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card" style="animation: fadeInUp 0.8s ease-out 0.2s both;">
                <h1 style="font-size: 4em; margin: 0; color: {risk_color}; font-weight: 800; background: none; -webkit-text-fill-color: unset;">{heart_prob*100:.1f}%</h1>
                <h2 style="color: {risk_color}; margin: 20px 0; font-size: 2em;">Probability</h2>
                <p style="font-size: 1.3em; color: #999; font-weight: 600;">Heart Disease Score</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Animated progress bar
        st.markdown('<h3 style="text-align: center; text-shadow: 0 2px 10px rgba(0,0,0,0.5); ">Risk Meter</h3>', unsafe_allow_html=True)
        st.markdown(f"""
        <div style="width: 100%; height: 20px; background-color: rgba(0, 91, 120, 0.53); border-radius: 15px; overflow: hidden; margin: 20px 0; border: 1px solid rgba(255,255,255,0.2);">
            <div style="width: {heart_prob*100}%; height: 100%; background-color: {risk_color}; border-radius: 15px; transition: width 0.8s ease-out; box-shadow: 0 0 20px rgba({risk_color}, 0.5);"></div>
        </div>
        """, unsafe_allow_html=True)
        
        # Recommendation with animation
        if heart_prob < 0.3:
            st.success(f"**💚 Recommendation:** {recommendation}")
        elif heart_prob < 0.6:
            st.warning(f"**⚡ Recommendation:** {recommendation}")
        else:
            st.error(f"**🚨 Recommendation:** {recommendation}")
        
        # Additional Info
        with st.expander("📊 View Detailed Probability Breakdown"):
            st.markdown(f"""
            <div style="padding: 20px;">
                <h4>Probability Distribution:</h4>
                <p style="font-size: 1.2em;">🟢 <strong>No Heart Disease:</strong> {proba[0]:.2%}</p>
                <p style="font-size: 1.2em;">🔴 <strong>Heart Disease:</strong> {proba[1]:.2%}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Log prediction
        log_data = input_df.copy()
        log_data["username"] = username if username else "Anonymous"
        log_data["prediction"] = prediction
        log_data["probability"] = heart_prob
        log_data["timestamp"] = pd.Timestamp.now()

        log_file = "prediction_logs.csv"
        if os.path.exists(log_file):
            log_data.to_csv(log_file, mode="a", header=False, index=False)
        else:
            log_data.to_csv(log_file, index=False)

        st.info("📁 Prediction logged successfully.")

    except Exception as e:
        st.error(f"❌ Error during prediction: {e}")
        st.info("Please ensure all inputs are valid and try again.")

# =========================
# FOOTER
# =========================
st.markdown("---")
st.markdown("""
<div style="text-align: center;  padding: 20px; text-shadow: 0 2px 10px rgba(0,0,0,0.5);">
    <p style="font-size: 1.2em;"><strong>Heart Disease Prediction System</strong> | Powered by Machine Learning ⚡</p>
    <p style="font-size: 1em;">⚠️ This tool is for educational purposes only. Always consult healthcare professionals for medical advice.</p>
    <p style="font-size: 0.9em;">© 2025 | Built with ❤️ using Streamlit & AI</p>
</div>
""", unsafe_allow_html=True)
