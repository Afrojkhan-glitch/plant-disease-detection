import streamlit as st
import tensorflow as tf
import numpy as np

# =========================
# LOAD MODEL (CACHE)
# =========================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model.keras",compile=False)

model = load_model()

# =========================
# CLASS NAMES (YOUR DATASET)
# =========================
class_names = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
               'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew',
               'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
               'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
               'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
               'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
               'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
               'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
               'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
               'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot',
               'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
               'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
               'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
               'Tomato___Tomato_mosaic_virus', 'Tomato___healthy']

# =========================
# DISEASE KNOWLEDGE BASE (IMPORTANT CROPS ONLY)
# =========================
disease_info = {

    # 🍅 TOMATO
    "Tomato___Early_blight": {
        "info": "Fungal disease causing brown spots on leaves.",
        "treatment": "Use Mancozeb or copper fungicide.",
        "prevention": "Avoid wet leaves and ensure spacing.",
        "advice": "Remove infected leaves immediately."
    },

    "Tomato___Late_blight": {
        "info": "Highly destructive fungal disease.",
        "treatment": "Use chlorothalonil or fungicide spray.",
        "prevention": "Improve air circulation.",
        "advice": "Destroy infected plants quickly."
    },

    "Tomato___healthy": {
        "info": "Healthy tomato plant.",
        "treatment": "No treatment needed.",
        "prevention": "Maintain proper farming practices.",
        "advice": "Continue monitoring."
    },

    # 🥔 POTATO
    "Potato___Early_blight": {
        "info": "Fungal disease causing leaf spots.",
        "treatment": "Use fungicide spray.",
        "prevention": "Crop rotation and hygiene.",
        "advice": "Remove infected leaves."
    },

    "Potato___Late_blight": {
        "info": "Severe destructive disease.",
        "treatment": "Use fungicides immediately.",
        "prevention": "Avoid moisture buildup.",
        "advice": "Destroy infected plants."
    },

    # 🍎 APPLE
    "Apple___Apple_scab": {
        "info": "Fungal disease affecting leaves and fruits.",
        "treatment": "Use fungicides like captan.",
        "prevention": "Remove fallen leaves.",
        "advice": "Prune infected branches."
    },

    # 🌽 CORN
    "Corn_(maize)___Common_rust_": {
        "info": "Fungal rust disease on maize.",
        "treatment": "Use fungicides like azoxystrobin.",
        "prevention": "Use resistant seeds.",
        "advice": "Remove infected leaves."
    },

    # 🍇 GRAPE
    "Grape___Black_rot": {
        "info": "Fungal disease affecting grapes.",
        "treatment": "Use mancozeb spray.",
        "prevention": "Prune infected parts.",
        "advice": "Remove infected grapes."
    },

    # 🌾 RICE (ONLY IF PRESENT IN YOUR DATASET)
    "Rice___Leaf_blast": {
        "info": "Fungal disease in rice leaves.",
        "treatment": "Use tricyclazole fungicide.",
        "prevention": "Avoid excess nitrogen.",
        "advice": "Use resistant varieties."
    },

    # 🌾 WHEAT
    "Wheat___Yellow_rust": {
        "info": "Yellow rust disease in wheat.",
        "treatment": "Use propiconazole fungicide.",
        "prevention": "Plant resistant seeds.",
        "advice": "Monitor early symptoms."
    }
}

# =========================
# DEFAULT FALLBACK
# =========================
default_info = {
    "info": "Disease detected in plant leaf.",
    "treatment": "Use general fungicide or consult expert.",
    "prevention": "Maintain crop hygiene and irrigation.",
    "advice": "Monitor regularly and act early."
}

# =========================
# PREDICTION FUNCTION
# =========================
def predict(image):
    img = tf.keras.preprocessing.image.load_img(image, target_size=(128, 128))
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = np.expand_dims(img, axis=0)

    preds = model.predict(img)

    top3_idx = np.argsort(preds[0])[::-1][:3]
    top3_probs = preds[0][top3_idx]

    return top3_idx, top3_probs

# =========================
# MULTILANGUAGE CHATBOT
# =========================
def chatbot(disease, lang):

    responses = {
        "English": f"Disease: {disease}\n\n💊 Use fungicide, remove infected parts, and monitor regularly.",
        "Nepali": f"रोग: {disease}\n\n💊 संक्रमित भाग हटाउनुहोस्, औषधि प्रयोग गर्नुहोस्।",
        "Hindi": f"बीमारी: {disease}\n\n💊 संक्रमित भाग हटाएँ और दवा का उपयोग करें।"
    }

    return responses[lang]

# =========================
# UI
# =========================
st.title("🌿 Smart Plant Disease Detection AI")

language = st.selectbox("🌍 Select Language", ["English", "Nepali", "Hindi"])

uploaded_file = st.file_uploader("Upload Plant Leaf Image", type=["jpg", "png", "jpeg"])

if uploaded_file:

    st.image(uploaded_file, use_container_width=True)

    if st.button("Predict Disease 🚀"):

        top3_idx, top3_probs = predict(uploaded_file)

        result = class_names[top3_idx[0]]
        confidence = top3_probs[0]

        st.success(f"🌿 Disease: {result}")
        st.info(f"🔍 Confidence: {confidence * 100:.2f}%")

        if confidence < 0.60:
            st.warning("⚠ Low confidence prediction")

        # TOP 3
        st.subheader("🏆 Top Predictions")
        for i in range(3):
            st.write(f"{class_names[top3_idx[i]]} — {top3_probs[i]*100:.2f}%")

        # INFO SYSTEM
        info = disease_info.get(result, default_info)

        st.subheader("🌿 Information")
        st.write(info["info"])

        st.subheader("💊 Treatment")
        st.write(info["treatment"])

        st.subheader("🛡 Prevention")
        st.write(info["prevention"])

        st.subheader("🌾 Advice")
        st.write(info["advice"])

        # CHATBOT
        st.subheader("🤖 Farmer AI Chatbot")

        st.write(chatbot(result, language))