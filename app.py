
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import io

# Constants
IMG_HEIGHT = 224
IMG_WIDTH = 224
CLASS_NAMES = ["daisy", "dandelions", "roses", "sunflowers", "tulip"]

# Configure Streamlit page
st.set_page_config(
    page_title="Flower Classifier",
    page_icon="🌸",
    layout="centered"
)

# Cache the model loading to avoid reloading on every interaction
@st.cache_resource
def load_model():
    """Load the trained flower classification model"""
    try:
        model = tf.keras.models.load_model("shallow_model.keras")
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.error("Make sure 'shallow_model.keras' is in the same directory as this app.")
        return None

def preprocess_image(image):
    """Preprocess the uploaded image for prediction"""
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize image to model input size
    image = image.resize((IMG_WIDTH, IMG_HEIGHT))
    
    # Convert to numpy array
    img_array = np.array(image)
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

def predict_flower(model, image):
    """Make prediction on the preprocessed image"""
    try:
        # Get model predictions (logits)
        predictions = model.predict(image, verbose=0)
        
        # Apply softmax to get probabilities
        probabilities = tf.nn.softmax(predictions[0]).numpy()
        
        # Get predicted class
        predicted_class_idx = np.argmax(probabilities)
        predicted_class = CLASS_NAMES[predicted_class_idx]
        confidence = probabilities[predicted_class_idx]
        
        return predicted_class, confidence, probabilities
    
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        return None, None, None

# Main app
def main():
    st.title("🌸 Flower Classification App")
    st.write("Upload an image of a flower and I'll predict what type it is!")
    
    # Load model
    model = load_model()
    
    if model is None:
        st.stop()
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a flower image...", 
        type=['jpg', 'jpeg', 'png'],
        help="Upload a clear image of a flower for best results"
    )
    
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        
        # Create two columns for layout
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Uploaded Image")
            st.image(image, caption="Your flower image", use_container_width=True)
        
        with col2:
            st.subheader("Prediction Results")
            
            # Preprocess image
            processed_image = preprocess_image(image)
            
            # Make prediction
            with st.spinner("Analyzing the flower..."):
                predicted_class, confidence, probabilities = predict_flower(model, processed_image)
            
            if predicted_class is not None:
                # Display main prediction
                st.success(f"**Prediction: {predicted_class.title()}**")
                st.write(f"**Confidence: {confidence:.1%}**")
                
                # Show confidence level with color coding
                if confidence > 0.7:
                    st.success("🎯 High confidence prediction!")
                elif confidence > 0.4:
                    st.warning("⚠️ Moderate confidence prediction")
                else:
                    st.error("❌ Low confidence prediction")
                
                # Display all class probabilities
                st.subheader("All Class Probabilities")
                for i, class_name in enumerate(CLASS_NAMES):
                    prob = float(probabilities[i])  # Convert to Python float
                    st.write(f"**{class_name.title()}:** {prob:.1%}")
                    st.progress(prob)
        
        # Additional information
        st.subheader("📋 Model Information")
        st.info("""
        This model is a simple neural network with:
        - **Architecture:** Single dense layer (no hidden layers)
        - **Input:** 224x224x3 RGB images
        - **Classes:** Daisy, Dandelions, Roses, Sunflowers, Tulip
        - **Note:** This is a shallow network without activation functions for educational purposes
        """)
        
        # Tips for better results
        st.subheader("💡 Tips for Better Results")
        st.write("""
        - Use clear, well-lit images
        - Make sure the flower is the main subject
        - Try different angles if the prediction confidence is low
        - The model works best with the 5 flower types it was trained on
        """)
    
    else:
        # Show example images and instructions
        st.subheader("📸 How to Use")
        st.write("""
        1. Click 'Browse files' above to upload a flower image
        2. The model will analyze your image and predict the flower type
        3. You'll see the prediction confidence and probabilities for all classes
        """)
        
        # Display supported flower types
        st.subheader("🌼 Supported Flower Types")
        cols = st.columns(5)
        for i, flower in enumerate(CLASS_NAMES):
            with cols[i]:
                st.write(f"**{flower.title()}**")

if __name__ == "__main__":
    main()