import streamlit as st
import tensorflow as tf
import numpy as np
import gdown
import os
from tensorflow.keras.layers import InputLayer, Rescaling, Normalization, ZeroPadding2D, Conv2D, Dense

# PAGE CONFIG

st.set_page_config(
    page_title="PlantAI — Disease Detection",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# CUSTOM CSS

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0d1117;
    color: #e6edf3;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1rem 1rem !important; max-width: 1100px !important; }
.hero {
    background: linear-gradient(135deg, #0a3d1f 0%, #0d5c2e 50%, #0a3d1f 100%);
    border: 1px solid #1a5c30;
    border-radius: 20px;
    padding: 2rem 1.5rem 1.5rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 220px; height: 220px;
    background: radial-gradient(circle, rgba(34,197,94,0.15) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: clamp(1.8rem, 5vw, 2.8rem);
    color: #4ade80;
    margin: 0 0 0.4rem;
    line-height: 1.15;
}
.hero-sub {
    font-size: clamp(0.85rem, 2.5vw, 1rem);
    color: #86efac;
    font-weight: 300;
    letter-spacing: 0.04em;
}
.hero-badge {
    display: inline-block;
    background: rgba(74,222,128,0.15);
    border: 1px solid #4ade80;
    color: #4ade80;
    font-size: 0.72rem;
    font-weight: 500;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    padding: 4px 12px;
    border-radius: 20px;
    margin-bottom: 1rem;
}
.card {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 14px;
    padding: 1.2rem;
    margin-bottom: 1rem;
}
.result-card {
    background: linear-gradient(135deg, #0d2818, #0a3d1f);
    border: 1px solid #22c55e;
    border-radius: 14px;
    padding: 1.2rem 1.5rem;
    margin: 1rem 0;
}
.result-disease {
    font-family: 'Playfair Display', serif;
    font-size: clamp(1.2rem, 4vw, 1.6rem);
    color: #4ade80;
    margin: 0 0 0.3rem;
    word-break: break-word;
}
.result-conf { font-size: 0.9rem; color: #86efac; font-weight: 300; }
.info-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.8rem;
    margin-top: 1rem;
}
@media (max-width: 600px) {
    .info-grid { grid-template-columns: 1fr !important; }
}
.info-item {
    background: #0d1117;
    border: 1px solid #21262d;
    border-radius: 10px;
    padding: 0.9rem 1rem;
}
.info-label {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #4ade80;
    font-weight: 500;
    margin-bottom: 0.4rem;
}
.info-text { font-size: 0.88rem; color: #c9d1d9; line-height: 1.5; }
.pred-bar-wrap { margin: 0.5rem 0; }
.pred-label { font-size: 0.82rem; color: #c9d1d9; margin-bottom: 3px; word-break: break-word; }
.pred-bar-bg { background: #21262d; border-radius: 6px; height: 8px; overflow: hidden; }
.pred-bar-fill { height: 100%; border-radius: 6px; background: linear-gradient(90deg, #166534, #4ade80); }
.pred-pct { font-size: 0.75rem; color: #4ade80; margin-top: 2px; text-align: right; }
.chat-bubble {
    background: #0d2818;
    border: 1px solid #166534;
    border-radius: 12px 12px 12px 2px;
    padding: 1rem 1.2rem;
    font-size: 0.9rem;
    color: #c9d1d9;
    line-height: 1.6;
    margin-top: 0.5rem;
    word-break: break-word;
}
.chat-icon {
    width: 32px; height: 32px;
    background: #166534;
    border-radius: 50%;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-size: 1rem;
    margin-bottom: 0.5rem;
    flex-shrink: 0;
}
.warn-box {
    background: #2d1b00;
    border: 1px solid #d97706;
    border-radius: 10px;
    padding: 0.8rem 1.2rem;
    color: #fbbf24;
    font-size: 0.88rem;
    margin: 0.8rem 0;
}
.error-box {
    background: #2d0000;
    border: 1px solid #ef4444;
    border-radius: 10px;
    padding: 0.8rem 1.2rem;
    color: #fca5a5;
    font-size: 0.88rem;
    margin: 0.8rem 0;
}
.stat-row { display: flex; gap: 0.6rem; margin-top: 1rem; flex-wrap: wrap; }
.stat-pill {
    background: rgba(74,222,128,0.08);
    border: 1px solid rgba(74,222,128,0.2);
    border-radius: 8px;
    padding: 0.4rem 0.8rem;
    font-size: clamp(0.72rem, 2vw, 0.82rem);
    color: #86efac;
}
.stat-pill strong { color: #4ade80; }
.divider { border: none; border-top: 1px solid #21262d; margin: 1.5rem 0; }
.stButton > button {
    background: linear-gradient(135deg, #166534, #15803d) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.6rem 2rem !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    font-size: 1rem !important;
    width: 100% !important;
}
.stButton > button:hover { opacity: 0.85 !important; }
.sec-heading {
    font-family: 'Playfair Display', serif;
    font-size: 1.1rem;
    color: #4ade80;
    margin: 1.2rem 0 0.6rem;
}
@media (max-width: 768px) {
    .block-container { padding: 0.5rem 0.5rem !important; }
    .hero { padding: 1.5rem 1rem 1rem; border-radius: 14px; }
    .result-card { padding: 1rem; }
}
[data-testid="stFileUploader"] { width: 100% !important; }
[data-testid="stSelectbox"] { width: 100% !important; }
[data-testid="stImage"] img { border-radius: 12px !important; width: 100% !important; }
</style>
""", unsafe_allow_html=True)


# LOAD MODEL


def create_legacy_wrapper(LayerClass):
    class LegacyLayer(LayerClass):
        def __init__(self, *args, **kwargs):
            for key in ['batch_shape', 'shape', 'dtype_policy', 'batch_input_shape', 'dtype']:
                kwargs.pop(key, None)
            super().__init__(*args, **kwargs)
    return LegacyLayer

@st.cache_resource
def load_model():
    model_path = "model.h5"
    file_id = "1bIHaucVRm66zzIcEckbYGbLe_WRrebW7"
    url = f'https://drive.google.com/uc?id={file_id}'
    
    if not os.path.exists(model_path):
        with st.spinner("Downloading Model from Drive..."):
            gdown.download(url, model_path, quiet=False)
    
    try:
        custom_map = {
            'InputLayer': create_legacy_wrapper(InputLayer),
            'Rescaling': create_legacy_wrapper(Rescaling),
            'Normalization': create_legacy_wrapper(Normalization),
            'ZeroPadding2D': create_legacy_wrapper(ZeroPadding2D),
            'Conv2D': create_legacy_wrapper(Conv2D),
            'Dense': create_legacy_wrapper(Dense)
        }
        
        return tf.keras.models.load_model(
            model_path, 
            custom_objects=custom_map, 
            compile=False,
            safe_mode=False
        )
    except Exception as e:
        st.error(f"Deep Load Failure: {e}")
        return None

# --- EXECUTION ---
model = load_model()

if model is None:
    st.error("Model failed to load. Please Delete and Re-deploy the app to clear the cache.")
    st.stop()
else:
    st.success("Success! The entire model architecture has been bridged and loaded.")
    
# CLASS NAMES

class_names = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Cherry__Spot_Leaf', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy', 'Orange___Huanglongbing_(Citrus_greening)', 'Peach__Rust_Leaf',
    'Peach___Bacterial_spot', 'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
    'Potato__Mosaic_virus', 'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
    'Raspberry__Leaf_spot', 'Raspberry__Rust_Leaf', 'Raspberry___healthy',
    'Rice__Bacterial_Leaf_Blight', 'Rice__Brown_Spot', 'Rice__Leaf_Blast', 'Rice__Leaf_Scald',
    'Rice__Narrow_Brown_Spot', 'Rice__Tungro', 'Rice__healthy',
    'Soybean__Bacterial_blight', 'Soybean__Frogeye_Leaf', 'Soybean__Rust_Leaf',
    'Soybean__Septoria_Brown_spot', 'Soybean___healthy', 'Squash___Powdery_mildew',
    'Strawberry__Leaf_spot', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy', 'Wheat__Brown_Rust', 'Wheat__Loose_Smut', 'Wheat__Septoria_Leaf',
    'Wheat__Yellow_Rust', 'Wheat__healthy_Leaf'
]


# DISEASE KNOWLEDGE BASE

disease_info = {
    "Apple___Apple_scab": {"info": "Fungal disease causing dark scabby lesions on leaves and fruits.", "treatment": "Apply fungicides like captan or myclobutanil.", "prevention": "Remove fallen leaves, improve air circulation.", "advice": "Prune infected branches and monitor regularly."},
    "Apple___Black_rot": {"info": "Fungal disease causing black rotting on fruits and leaves.", "treatment": "Use copper-based fungicide spray.", "prevention": "Remove mummified fruits and dead branches.", "advice": "Sanitize pruning tools between cuts."},
    "Apple___Cedar_apple_rust": {"info": "Fungal disease causing orange spots on apple leaves.", "treatment": "Apply fungicides at bud break stage.", "prevention": "Remove nearby cedar trees if possible.", "advice": "Monitor during wet spring weather."},
    "Apple___healthy": {"info": "Healthy apple plant.", "treatment": "No treatment needed.", "prevention": "Maintain proper pruning and fertilization.", "advice": "Continue regular monitoring."},
    "Blueberry___healthy": {"info": "Healthy blueberry plant.", "treatment": "No treatment needed.", "prevention": "Maintain soil acidity and proper watering.", "advice": "Monitor for pests regularly."},
    "Cherry_(including_sour)___Powdery_mildew": {"info": "Fungal disease causing white powdery coating on leaves.", "treatment": "Use sulfur-based or potassium bicarbonate fungicide.", "prevention": "Ensure good air circulation around plants.", "advice": "Avoid overhead irrigation."},
    "Cherry_(including_sour)___healthy": {"info": "Healthy cherry plant.", "treatment": "No treatment needed.", "prevention": "Regular pruning and monitoring.", "advice": "Check for pests seasonally."},
    "Cherry__Spot_Leaf": {"info": "Fungal disease causing reddish-brown spots on cherry leaves.", "treatment": "Apply copper fungicide or chlorothalonil.", "prevention": "Remove infected leaves promptly.", "advice": "Avoid wetting leaves during irrigation."},
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": {"info": "Fungal disease causing gray rectangular lesions on corn leaves.", "treatment": "Apply strobilurin fungicides.", "prevention": "Use resistant hybrids and practice crop rotation.", "advice": "Monitor fields during humid conditions."},
    "Corn_(maize)___Common_rust_": {"info": "Fungal rust disease causing brown pustules on maize leaves.", "treatment": "Use fungicides like azoxystrobin.", "prevention": "Plant resistant seed varieties.", "advice": "Scout fields early in the season."},
    "Corn_(maize)___Northern_Leaf_Blight": {"info": "Fungal disease causing long tan lesions on corn leaves.", "treatment": "Apply propiconazole fungicide.", "prevention": "Use resistant hybrids and rotate crops.", "advice": "Remove crop debris after harvest."},
    "Corn_(maize)___healthy": {"info": "Healthy maize plant.", "treatment": "No treatment needed.", "prevention": "Maintain proper spacing and fertilization.", "advice": "Continue monitoring for early signs."},
    "Grape___Black_rot": {"info": "Fungal disease causing black shriveled fruits and leaf spots.", "treatment": "Apply mancozeb or myclobutanil spray.", "prevention": "Prune and remove infected plant material.", "advice": "Spray preventively during wet weather."},
    "Grape___Esca_(Black_Measles)": {"info": "Complex fungal disease causing tiger-stripe patterns on leaves.", "treatment": "No fully effective chemical treatment; remove infected wood.", "prevention": "Protect pruning wounds with fungicide paste.", "advice": "Avoid water stress in vines."},
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {"info": "Fungal disease causing dark brown spots on grape leaves.", "treatment": "Use copper-based fungicides.", "prevention": "Remove infected leaves and improve airflow.", "advice": "Apply fungicide before rainy seasons."},
    "Grape___healthy": {"info": "Healthy grape vine.", "treatment": "No treatment needed.", "prevention": "Regular pruning and canopy management.", "advice": "Monitor for powdery mildew in summer."},
    "Orange___Huanglongbing_(Citrus_greening)": {"info": "Bacterial disease spread by insects causing yellowing and fruit drop.", "treatment": "No cure; remove infected trees to prevent spread.", "prevention": "Control psyllid insect vector with insecticides.", "advice": "Use certified disease-free planting material."},
    "Peach__Rust_Leaf": {"info": "Fungal rust disease causing orange pustules on peach leaves.", "treatment": "Apply sulfur or copper fungicide.", "prevention": "Remove infected leaves and nearby alternate hosts.", "advice": "Monitor during warm humid conditions."},
    "Peach___Bacterial_spot": {"info": "Bacterial disease causing dark spots on leaves and fruits.", "treatment": "Apply copper-based bactericide sprays.", "prevention": "Avoid overhead irrigation, plant resistant varieties.", "advice": "Prune to improve air circulation."},
    "Peach___healthy": {"info": "Healthy peach plant.", "treatment": "No treatment needed.", "prevention": "Regular pruning and pest monitoring.", "advice": "Fertilize appropriately each season."},
    "Pepper,_bell___Bacterial_spot": {"info": "Bacterial disease causing water-soaked spots on leaves and fruits.", "treatment": "Use copper-based bactericide.", "prevention": "Use disease-free seeds and avoid leaf wetness.", "advice": "Rotate crops and remove infected plants."},
    "Pepper,_bell___healthy": {"info": "Healthy bell pepper plant.", "treatment": "No treatment needed.", "prevention": "Ensure proper watering and drainage.", "advice": "Monitor for aphids and spider mites."},
    "Potato__Mosaic_virus": {"info": "Viral disease causing mosaic patterns and leaf distortion.", "treatment": "No cure; remove and destroy infected plants.", "prevention": "Use virus-free certified seed potatoes.", "advice": "Control aphid vectors with insecticide."},
    "Potato___Early_blight": {"info": "Fungal disease causing dark brown spots with concentric rings.", "treatment": "Apply chlorothalonil or mancozeb fungicide.", "prevention": "Crop rotation and proper plant spacing.", "advice": "Remove infected leaves promptly."},
    "Potato___Late_blight": {"info": "Devastating fungal disease causing dark water-soaked lesions.", "treatment": "Use fungicides like metalaxyl immediately.", "prevention": "Avoid excess moisture and use resistant varieties.", "advice": "Destroy infected plants quickly to stop spread."},
    "Potato___healthy": {"info": "Healthy potato plant.", "treatment": "No treatment needed.", "prevention": "Use certified disease-free seed tubers.", "advice": "Monitor regularly for early symptoms."},
    "Raspberry__Leaf_spot": {"info": "Fungal disease causing circular spots on raspberry leaves.", "treatment": "Apply copper fungicide or captan.", "prevention": "Improve drainage and air circulation.", "advice": "Remove fallen infected leaves."},
    "Raspberry__Rust_Leaf": {"info": "Fungal rust causing yellow-orange pustules on leaves.", "treatment": "Apply sulfur or myclobutanil fungicide.", "prevention": "Remove infected canes and improve airflow.", "advice": "Avoid wetting foliage during watering."},
    "Raspberry___healthy": {"info": "Healthy raspberry plant.", "treatment": "No treatment needed.", "prevention": "Annual pruning and cane management.", "advice": "Monitor for botrytis during fruiting."},
    "Rice__Bacterial_Leaf_Blight": {"info": "Bacterial disease causing water-soaked lesions on rice leaves.", "treatment": "Apply copper bactericide; use resistant varieties.", "prevention": "Avoid excess nitrogen fertilizer.", "advice": "Drain fields during severe outbreaks."},
    "Rice__Brown_Spot": {"info": "Fungal disease causing brown oval spots on rice leaves.", "treatment": "Use mancozeb or iprodione fungicide.", "prevention": "Balanced fertilization and proper water management.", "advice": "Use certified disease-free seeds."},
    "Rice__Leaf_Blast": {"info": "Destructive fungal disease causing diamond-shaped lesions.", "treatment": "Apply tricyclazole or isoprothiolane fungicide.", "prevention": "Avoid excess nitrogen, use resistant varieties.", "advice": "Monitor during cool humid weather."},
    "Rice__Leaf_Scald": {"info": "Fungal disease causing scalded appearance on rice leaves.", "treatment": "Apply propiconazole fungicide.", "prevention": "Balanced soil nutrition and good drainage.", "advice": "Remove infected plant debris after harvest."},
    "Rice__Narrow_Brown_Spot": {"info": "Fungal disease causing narrow brown streaks on rice leaves.", "treatment": "Use mancozeb spray.", "prevention": "Proper fertilization and water management.", "advice": "Use resistant rice varieties."},
    "Rice__Tungro": {"info": "Viral disease causing yellow-orange discoloration in rice.", "treatment": "No cure; remove infected plants immediately.", "prevention": "Control green leafhopper vector with insecticide.", "advice": "Plant tungro-resistant varieties."},
    "Rice__healthy": {"info": "Healthy rice plant.", "treatment": "No treatment needed.", "prevention": "Maintain proper water levels and fertilization.", "advice": "Scout regularly for leafhopper insects."},
    "Soybean__Bacterial_blight": {"info": "Bacterial disease causing water-soaked angular leaf spots.", "treatment": "Apply copper-based bactericide.", "prevention": "Use disease-free seeds and avoid leaf wetness.", "advice": "Practice crop rotation with non-legumes."},
    "Soybean__Frogeye_Leaf": {"info": "Fungal disease causing circular gray spots with brown borders.", "treatment": "Apply azoxystrobin or pyraclostrobin fungicide.", "prevention": "Use resistant varieties and rotate crops.", "advice": "Scout fields during humid conditions."},
    "Soybean__Rust_Leaf": {"info": "Aggressive fungal rust causing tan lesions and defoliation.", "treatment": "Apply triazole fungicides at first sign.", "prevention": "Early planting and use of resistant varieties.", "advice": "Monitor closely — spreads rapidly."},
    "Soybean__Septoria_Brown_spot": {"info": "Fungal disease causing brown irregular spots on lower leaves.", "treatment": "Use thiophanate-methyl fungicide.", "prevention": "Crop rotation and residue management.", "advice": "Apply fungicide at early pod stage."},
    "Soybean___healthy": {"info": "Healthy soybean plant.", "treatment": "No treatment needed.", "prevention": "Proper spacing and weed management.", "advice": "Monitor for aphids and bean beetles."},
    "Squash___Powdery_mildew": {"info": "Fungal disease causing white powdery coating on squash leaves.", "treatment": "Apply potassium bicarbonate or sulfur fungicide.", "prevention": "Ensure good air circulation and avoid overhead watering.", "advice": "Remove heavily infected leaves."},
    "Strawberry__Leaf_spot": {"info": "Fungal disease causing purple-bordered spots on strawberry leaves.", "treatment": "Apply captan or copper fungicide.", "prevention": "Remove infected leaves and improve drainage.", "advice": "Avoid excessive nitrogen fertilizer."},
    "Strawberry___Leaf_scorch": {"info": "Fungal disease causing purple spots that dry and scorch leaves.", "treatment": "Use myclobutanil or copper fungicide.", "prevention": "Remove old leaves after harvest.", "advice": "Plant resistant varieties where available."},
    "Strawberry___healthy": {"info": "Healthy strawberry plant.", "treatment": "No treatment needed.", "prevention": "Regular renovation and weed control.", "advice": "Monitor for slugs and spider mites."},
    "Tomato___Bacterial_spot": {"info": "Bacterial disease causing small dark spots on leaves and fruits.", "treatment": "Use copper-based bactericide sprays.", "prevention": "Use disease-free seeds and avoid leaf wetness.", "advice": "Remove and destroy infected plant debris."},
    "Tomato___Early_blight": {"info": "Fungal disease causing brown spots with concentric rings.", "treatment": "Apply mancozeb or copper fungicide.", "prevention": "Avoid wet leaves and ensure plant spacing.", "advice": "Remove infected lower leaves immediately."},
    "Tomato___Late_blight": {"info": "Highly destructive fungal disease causing dark water-soaked lesions.", "treatment": "Use chlorothalonil or metalaxyl fungicide.", "prevention": "Improve air circulation and avoid overhead watering.", "advice": "Destroy infected plants quickly to prevent spread."},
    "Tomato___Leaf_Mold": {"info": "Fungal disease causing yellow patches and olive-green mold on leaves.", "treatment": "Apply chlorothalonil or mancozeb.", "prevention": "Reduce humidity, improve ventilation.", "advice": "Remove infected leaves immediately."},
    "Tomato___Septoria_leaf_spot": {"info": "Fungal disease causing small circular spots with dark borders.", "treatment": "Apply copper fungicide or chlorothalonil.", "prevention": "Avoid splashing water on leaves.", "advice": "Stake plants to improve air circulation."},
    "Tomato___Spider_mites Two-spotted_spider_mite": {"info": "Pest infestation causing yellow stippling and webbing on leaves.", "treatment": "Apply miticide or insecticidal soap.", "prevention": "Keep plants well-watered; avoid dry dusty conditions.", "advice": "Introduce natural predators like lacewings."},
    "Tomato___Target_Spot": {"info": "Fungal disease causing concentric ring spots on leaves.", "treatment": "Apply azoxystrobin or chlorothalonil.", "prevention": "Improve plant spacing and air circulation.", "advice": "Remove infected plant debris promptly."},
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {"info": "Viral disease spread by whiteflies causing leaf curling and yellowing.", "treatment": "No cure; remove infected plants.", "prevention": "Control whitefly populations with insecticide.", "advice": "Use reflective mulches to repel whiteflies."},
    "Tomato___Tomato_mosaic_virus": {"info": "Viral disease causing mosaic patterns and distorted leaves.", "treatment": "No cure; remove and destroy infected plants.", "prevention": "Use virus-free seeds and sanitize tools.", "advice": "Avoid smoking near plants — tobacco carries the virus."},
    "Tomato___healthy": {"info": "Healthy tomato plant.", "treatment": "No treatment needed.", "prevention": "Maintain proper watering and fertilization.", "advice": "Monitor regularly for early signs of disease."},
    "Wheat__Brown_Rust": {"info": "Fungal rust causing brown pustules on wheat leaves.", "treatment": "Apply propiconazole or tebuconazole fungicide.", "prevention": "Plant resistant varieties and monitor early.", "advice": "Apply fungicide at flag leaf stage."},
    "Wheat__Loose_Smut": {"info": "Fungal disease replacing grain with black smut spores.", "treatment": "Use systemic seed treatment fungicide.", "prevention": "Use certified smut-free seeds.", "advice": "Remove smutted heads before spore release."},
    "Wheat__Septoria_Leaf": {"info": "Fungal disease causing tan blotches on wheat leaves.", "treatment": "Apply azoxystrobin or propiconazole.", "prevention": "Rotate crops and use resistant varieties.", "advice": "Scout at tillering stage."},
    "Wheat__Yellow_Rust": {"info": "Fungal disease causing yellow stripe pustules on wheat leaves.", "treatment": "Apply propiconazole or mancozeb fungicide.", "prevention": "Plant resistant seed varieties.", "advice": "Monitor in cool moist conditions — spreads fast."},
    "Wheat__healthy_Leaf": {"info": "Healthy wheat plant.", "treatment": "No treatment needed.", "prevention": "Proper fertilization and weed control.", "advice": "Monitor for aphids and rust during growing season."},
}

default_info = {
    "info": "Disease detected in plant leaf.",
    "treatment": "Use general fungicide or consult an agricultural expert.",
    "prevention": "Maintain crop hygiene and proper irrigation.",
    "advice": "Monitor regularly and act early for best results."
}


# PREDICTION

def predict(image):
    img = tf.keras.preprocessing.image.load_img(image, target_size=(224, 224))
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    preds = model.predict(img)
    top3_idx = np.argsort(preds[0])[::-1][:3]
    top3_probs = preds[0][top3_idx]
    return top3_idx, top3_probs, preds[0]


# VALIDATE PREDICTION

def is_valid_prediction(all_preds):
    sorted_preds = np.sort(all_preds)[::-1]
    top1 = sorted_preds[0]
    top2 = sorted_preds[1]
    gap  = top1 - top2
    # Car → gap > 90% → REJECT
    # Real leaf → gap < 90% → ACCEPT
    if gap > 0.90:
        return False
    return True


# CHATBOT

def chatbot(disease, lang):
    clean = disease.replace("___", " → ").replace("__", " → ").replace("_", " ")
    is_healthy = "healthy" in disease.lower()
    if is_healthy:
        responses = {
            "English": f"<b>Diagnosis:</b> {clean}<br><br>Your plant appears to be healthy! No disease was detected. Continue with regular watering, proper fertilization, and routine monitoring. Ensure adequate sunlight and good air circulation to keep your plant thriving.",
            "Nepali":  f"<b>निदान:</b> {clean}<br><br>तपाईंको बिरुवा स्वस्थ देखिन्छ! कुनै रोग फेला परेन। नियमित सिँचाइ, उचित मलखाद र नियमित निगरानी जारी राख्नुहोस्।",
            "Hindi":   f"<b>निदान:</b> {clean}<br><br>आपका पौधा स्वस्थ दिखता है! कोई बीमारी नहीं मिली। नियमित सिंचाई, उचित उर्वरक और नियमित निगरानी जारी रखें।"
        }
    else:
        responses = {
            "English": f"<b>Diagnosis:</b> {clean}<br><br>Apply the recommended fungicide early in the morning. Remove and dispose of all infected plant material away from the field. Ensure adequate spacing between plants for airflow. Continue monitoring every 3–5 days and consult your local agricultural officer if symptoms worsen.",
            "Nepali":  f"<b>रोग:</b> {clean}<br><br>सिफारिश गरिएको ढुसीनाशक बिहान छिटो प्रयोग गर्नुहोस्। संक्रमित बिरुवाको भाग खेतबाट टाढा राखी नष्ट गर्नुहोस्। हरेक ३–५ दिनमा निगरानी गर्नुहोस्।",
            "Hindi":   f"<b>बीमारी:</b> {clean}<br><br>सुबह जल्दी अनुशंसित कवकनाशी लगाएं। संक्रमित पौधे सामग्री को खेत से दूर हटाएं। हर 3–5 दिनों में निगरानी करें।"
        }
    return responses.get(lang, responses["English"])


# HERO

st.markdown("""
<div class="hero">
    <div class="hero-badge">AI-Powered · Final Year Project</div>
    <div class="hero-title">🌿 PlantAI</div>
    <div class="hero-sub">Smart Plant Disease Detection & Advisory System</div>
    <div class="stat-row">
        <div class="stat-pill"><strong>60</strong> Disease Classes</div>
        <div class="stat-pill"><strong>EfficientNet</strong> Architecture</div>
        <div class="stat-pill"><strong>224×224</strong> Resolution</div>
        <div class="stat-pill"><strong>3</strong> Languages</div>
    </div>
</div>
""", unsafe_allow_html=True)


# LAYOUT

col1, col2 = st.columns([1, 1.2], gap="large")

with col1:
    st.markdown('<div class="sec-heading">🌍 Language</div>', unsafe_allow_html=True)
    language = st.selectbox("", ["English", "Nepali", "Hindi"],
                            label_visibility="collapsed")

    st.markdown('<div class="sec-heading">📷 Upload Leaf Image</div>',
                unsafe_allow_html=True)
    uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"],
                                     label_visibility="collapsed")

    if uploaded_file:
        st.image(uploaded_file, use_container_width=True,
                 caption="Uploaded leaf image")
        predict_btn = st.button("Analyze Disease")
    else:
        st.markdown("""
        <div class="card" style="text-align:center; padding:2rem;">
            <div style="font-size:3rem">🍃</div>
            <div style="font-size:0.88rem; color:#6e7681; margin-top:0.5rem;">
                Upload a clear photo of a plant leaf to begin diagnosis
            </div>
        </div>
        """, unsafe_allow_html=True)
        predict_btn = False

with col2:
    if uploaded_file and predict_btn:
        with st.spinner("Analyzing leaf..."):
            top3_idx, top3_probs, all_preds = predict(uploaded_file)

        result     = class_names[top3_idx[0]]
        confidence = top3_probs[0]

        # ── Validate ─────────────────────────────────
        if not is_valid_prediction(all_preds):
            st.markdown("""
            <div class="error-box">
                This does not appear to be a plant leaf image.<br>
                Please upload a clear photo of a plant leaf.
            </div>
            """, unsafe_allow_html=True)
            st.stop()

        # ── Result card ──────────────────────────────
        clean_result = result.replace("___", " → ").replace("__", " → ").replace("_", " ")
        is_healthy   = "healthy" in result.lower()
        color        = "#4ade80" if is_healthy else "#f87171"
        icon         = "✅" if is_healthy else "⚠️"

        st.markdown(f"""
        <div class="result-card" style="border-color:{color};">
            <div style="font-size:0.75rem;text-transform:uppercase;
            letter-spacing:0.1em;color:{color};margin-bottom:0.5rem;">
                {icon} Diagnosis Result
            </div>
            <div class="result-disease" style="color:{color};">{clean_result}</div>
            <div class="result-conf">Confidence: {confidence * 100:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

        if confidence < 0.60:
            st.markdown("""
            <div class="warn-box">
            ⚠️ Low confidence — please upload a clearer, well-lit image.
            </div>
            """, unsafe_allow_html=True)

        # ── Top 3 ────────────────────────────────────
        st.markdown('<div class="sec-heading">Top 3 Predictions</div>',
                    unsafe_allow_html=True)
        for i in range(3):
            name = class_names[top3_idx[i]].replace(
                "___", " → ").replace("__", " → ").replace("_", " ")
            pct = top3_probs[i] * 100
            st.markdown(f"""
            <div class="pred-bar-wrap">
                <div class="pred-label">{name}</div>
                <div class="pred-bar-bg">
                    <div class="pred-bar-fill" style="width:{int(pct)}%"></div>
                </div>
                <div class="pred-pct">{pct:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

        # ── Disease info ─────────────────────────────
        info = disease_info.get(result, default_info)
        st.markdown('<div class="sec-heading">📋 Disease Information</div>',
                    unsafe_allow_html=True)
        st.markdown(f"""
        <div class="info-grid">
            <div class="info-item">
                <div class="info-label">ℹ️ About</div>
                <div class="info-text">{info["info"]}</div>
            </div>
            <div class="info-item">
                <div class="info-label">💊 Treatment</div>
                <div class="info-text">{info["treatment"]}</div>
            </div>
            <div class="info-item">
                <div class="info-label">🛡️ Prevention</div>
                <div class="info-text">{info["prevention"]}</div>
            </div>
            <div class="info-item">
                <div class="info-label">🌾 Advice</div>
                <div class="info-text">{info["advice"]}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Farmer AI Advisory ───────────────────────
        st.markdown('<div class="sec-heading">🌾 Farmer AI Advisory</div>',
                    unsafe_allow_html=True)
        st.markdown(f"""
        <div style="display:flex;align-items:flex-start;gap:0.8rem;">
            <div class="chat-icon">🌿</div>
            <div class="chat-bubble">{chatbot(result, language)}</div>
        </div>
        """, unsafe_allow_html=True)

    elif not uploaded_file:
        st.markdown("""
        <div class="card" style="margin-top:2.5rem;padding:2.5rem;text-align:center;">
            <div style="font-size:2.5rem;margin-bottom:1rem;">🔬</div>
            <div style="font-family:'Playfair Display',serif;font-size:1.2rem;
            color:#4ade80;margin-bottom:0.5rem;">How it works</div>
            <div style="color:#6e7681;font-size:0.88rem;line-height:1.8;">
                1. Upload a photo of a plant leaf<br>
                2. Click Analyze Disease<br>
                3. Get instant diagnosis + treatment advice<br>
                4. Available in English, Hindi & Nepali
            </div>
        </div>
        """, unsafe_allow_html=True)


# FOOTER

st.markdown('<hr class="divider">', unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center;color:#4d5562;font-size:0.78rem;padding-bottom:1rem;">
    PlantAI — AI Based Plant Disease Detection System · © 2026 Afroj Khan · All Rights Reserved
</div>
""", unsafe_allow_html=True)
