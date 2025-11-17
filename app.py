import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import time
import json

# Set page configuration
st.set_page_config(
    page_title="Dog vs Cat Classifier",
    page_icon="🐕",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        background: linear-gradient(45deg, #FF6B6B, #4ECDC4, #45B7D1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    .dog-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 25px;
        border-radius: 15px;
        color: white;
        margin: 10px 0;
        border: 3px solid #FF9A3D;
    }
    .cat-card {
        background: linear-gradient(135deg, #FF6B6B 0%, #FF8E53 100%);
        padding: 25px;
        border-radius: 15px;
        color: white;
        margin: 10px 0;
        border: 3px solid #4ECDC4;
    }
    .prediction-text {
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 10px;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
        text-align: center;
    }
    .confidence-text {
        font-size: 1.5rem;
        text-align: center;
    }
    .confidence-bar {
        background: rgba(255,255,255,0.3);
        border-radius: 10px;
        margin: 15px 0;
        overflow: hidden;
    }
    .confidence-fill {
        background: linear-gradient(90deg, #FFD93D, #FF6B6B);
        height: 30px;
        border-radius: 10px;
        text-align: center;
        color: black;
        font-weight: bold;
        line-height: 30px;
    }
    .stats-card {
        background: rgba(255,255,255,0.1);
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .feature-list {
        background: rgba(255,255,255,0.1);
        padding: 15px;
        border-radius: 10px;
        margin: 5px 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the pre-trained classifier"""
    try:
        model = keras.models.load_model('dog_cat_classifier.h5')
        st.success("✅ Dog vs Cat Classifier Loaded Successfully!")
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

@st.cache_data
def load_pet_info():
    """Load pet information"""
    try:
        with open('pet_info.json', 'r') as f:
            return json.load(f)
    except:
        return {}

def preprocess_image(image):
    """Preprocess the image for the model"""
    # Resize to 128x128 (model input size)
    image = image.resize((128, 128))
    img_array = np.array(image)
    
    # Ensure 3 channels
    if len(img_array.shape) == 2:  # Grayscale
        img_array = np.stack([img_array] * 3, axis=-1)
    elif img_array.shape[-1] == 4:  # RGBA
        img_array = img_array[:, :, :3]
    
    # Normalize
    img_array = img_array.astype('float32') / 255.0
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_pet(_model, image):
    """Predict if image contains Dog or Cat"""
    processed_image = preprocess_image(image)
    
    with st.spinner('🔍 Analyzing image...'):
        time.sleep(1.5)
        predictions = _model.predict(processed_image, verbose=0)
    
    # predictions[0] will be [dog_probability, cat_probability]
    dog_confidence = predictions[0][0]
    cat_confidence = predictions[0][1]
    
    return dog_confidence, cat_confidence

def main():
    st.markdown('<h1 class="main-header">🐕 Dog vs Cat Classifier</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    ### Deep Learning Pet Classification
    Upload an image and our AI will determine if it contains a **Dog** or **Cat** using Convolutional Neural Networks!
    """)
    
    # Load model and pet info
    pet_info = load_pet_info()
    model = load_model()
    
    if model is None:
        st.error("Model failed to load. Please check the model file.")
        return
    
    # Stats sidebar
    with st.sidebar:
        st.header("📊 Model Information")
        st.markdown("""
        <div class="stats-card">
            <strong>Task:</strong> Binary Classification<br>
            <strong>Classes:</strong> Dog 🐕 vs Cat 🐱<br>
            <strong>Input Size:</strong> 128×128 pixels<br>
            <strong>Architecture:</strong> Convolutional Neural Network<br>
            <strong>Training:</strong> Synthetic dataset
        </div>
        """, unsafe_allow_html=True)
        
        st.header("🎯 How It Works")
        st.info("""
        The AI analyzes visual patterns:
        - **Dogs**: Floppy ears, longer snouts
        - **Cats**: Pointy ears, slender faces
        - Uses deep learning feature extraction
        """)
        
        st.header("📈 Model Performance")
        st.write("**Accuracy:** ~85% on test data")
        st.write("**Framework:** TensorFlow/Keras")
        st.write("**Layers:** 3 Conv + 2 Dense")
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📸 Upload Pet Image")
        uploaded_file = st.file_uploader(
            "Choose an image of a Dog or Cat", 
            type=['jpg', 'jpeg', 'png'],
            help="Upload a clear image of a dog or cat"
        )
        
        image = None
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
    
    with col2:
        st.subheader("🔍 Classification Guide")
        
        st.markdown("""
        <div class="feature-list">
        <strong>🐕 Dog Features:</strong>
        • Floppy ears
        • Longer snout
        • Friendly expression
        • Various sizes and colors
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-list">
        <strong>🐱 Cat Features:</strong>
        • Pointy ears
        • Slender face
        • Graceful posture
        • Vertical pupils
        </div>
        """, unsafe_allow_html=True)
        
        st.subheader("💡 Tips")
        st.write("• Use clear, well-lit images")
        st.write("• Center the animal's face")
        st.write("• Avoid blurry images")
        st.write("• Good contrast helps")
    
    # Prediction
    if image is not None:
        st.markdown("---")
        st.subheader("🎯 Classification Results")
        
        dog_confidence, cat_confidence = predict_pet(model, image)
        
        # Determine prediction
        if dog_confidence > cat_confidence:
            prediction = "Dog"
            confidence = dog_confidence
            card_class = "dog-card"
            emoji = "🐕"
        else:
            prediction = "Cat"
            confidence = cat_confidence
            card_class = "cat-card"
            emoji = "🐱"
        
        # Display prediction
        st.markdown(f"""
        <div class="{card_class}">
            <div class="prediction-text">{emoji} {prediction}</div>
            <div class="confidence-text">Confidence: {confidence:.2%}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Confidence levels
        st.markdown("### 📊 Confidence Levels")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**🐕 Dog Confidence**")
            st.write(f"{dog_confidence:.2%}")
            progress_html = f"""
            <div class="confidence-bar">
                <div class="confidence-fill" style="width: {dog_confidence*100}%">
                    {dog_confidence:.2%}
                </div>
            </div>
            """
            st.markdown(progress_html, unsafe_allow_html=True)
        
        with col2:
            st.write("**🐱 Cat Confidence**")
            st.write(f"{cat_confidence:.2%}")
            progress_html = f"""
            <div class="confidence-bar">
                <div class="confidence-fill" style="width: {cat_confidence*100}%">
                    {cat_confidence:.2%}
                </div>
            </div>
            """
            st.markdown(progress_html, unsafe_allow_html=True)
        
        # Pet information
        with st.expander("📖 Pet Information"):
            if prediction in pet_info:
                info = pet_info[prediction]
                st.write(f"### About {prediction}s")
                st.write(f"**Description:** {info.get('description', 'N/A')}")
                
                st.write("**Characteristics:**")
                for feature in info.get('characteristics', []):
                    st.write(f"- {feature}")
                
                st.write(f"**Fun Fact:** {info.get('fun_fact', 'N/A')}")
                st.write(f"**Average Lifespan:** {info.get('lifespan', 'N/A')}")

if __name__ == "__main__":
    main()
