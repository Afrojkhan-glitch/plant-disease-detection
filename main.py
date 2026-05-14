import streamlit as st
import tensorflow as tf
import numpy as np
import gdown
import os
from tensorflow.keras.applications.efficientnet import preprocess_input


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
.block-container { padding: 2rem 3rem; max-width: 1100px; }

.hero {
    background: linear-gradient(135deg, #0a3d1f 0%, #0d5c2e 50%, #0a3d1f 100%);
    border: 1px solid #1a5c30;
    border-radius: 20px;
    padding: 3rem 2.5rem 2rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: 2.8rem;
    color: #4ade80;
    margin: 0 0 0.4rem;
    line-height: 1.15;
}
.hero-sub { font-size: 1rem; color: #86efac; font-weight: 300; letter-spacing: 0.04em; }
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
.card { background: #161b22; border: 1px solid #21262d; border-radius: 14px; padding: 1.5rem; margin-bottom: 1rem; }
.result-card { background: linear-gradient(135deg, #0d2818, #0a3d1f); border: 1px solid #22c55e; border-radius: 14px; padding: 1.5rem 2rem; margin: 1rem 0; }
.result-disease { font-family: 'Playfair Display', serif; font-size: 1.6rem; color: #4ade80; margin: 0 0 0.3rem; }
.result-conf { font-size: 0.9rem; color: #86efac; font-weight: 300; }
.info-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-top: 1rem; }
.info-item { background: #0d1117; border: 1px solid #21262d; border-radius: 10px; padding: 1rem 1.2rem; }
.info-label { font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.12em; color: #4ade80; font-weight: 500; margin-bottom: 0.4rem; }
.info-text { font-size: 0.92rem; color: #c9d1d9; line-height: 1.5; }
.pred-bar-wrap { margin: 0.5rem 0; }
.pred-label { font-size: 0.82rem; color: #c9d1d9; margin-bottom: 3px; }
.pred-bar-bg { background: #21262d; border-radius: 6px; height: 8px; overflow: hidden; }
.pred-bar-fill { height: 100%; border-radius: 6px; background: linear-gradient(90deg, #166534, #4ade80); }
.pred-pct { font-size: 0.75rem; color: #4ade80; margin-top: 2px; text-align: right; }
.chat-bubble { background: #0d2818; border: 1px solid #166534; border-radius: 12px 12px 12px 2px; padding: 1rem 1.2rem; font-size: 0.92rem; color: #c9d1d9; line-height: 1.6; margin-top: 0.5rem; }
.chat-icon { width: 32px; height: 32px; background: #166534; border-radius: 50%; display: inline-flex; align-items: center; justify-content: center; font-size: 1rem; margin-bottom: 0.5rem; }
.warn-box { background: #2d1b00; border: 1px solid #d97706; border-radius: 10px; padding: 0.8rem 1.2rem; color: #fbbf24; font-size: 0.88rem; margin: 0.8rem 0; }
.error-box { background: #2d0a0a; border: 2px solid #ef4444; border-radius: 14px; padding: 2rem; text-align: center; margin: 1rem 0; }
.error-icon { font-size: 3rem; margin-bottom: 0.8rem; }
.error-title { font-family: 'Playfair Display', serif; font-size: 1.4rem; color: #ef4444; margin-bottom: 0.5rem; }
.error-text { font-size: 0.9rem; color: #fca5a5; line-height: 1.6; }
.stat-row { display: flex; gap: 1rem; margin-top: 1rem; flex-wrap: wrap; }
.stat-pill { background: rgba(74,222,128,0.08); border: 1px solid rgba(74,222,128,0.2); border-radius: 8px; padding: 0.5rem 1rem; font-size: 0.82rem; color: #86efac; }
.stat-pill strong { color: #4ade80; }
.divider { border: none; border-top: 1px solid #21262d; margin: 1.5rem 0; }
.stButton > button { background: linear-gradient(135deg, #166534, #15803d) !important; color: white !important; border: none !important; border-radius: 10px !important; padding: 0.6rem 2rem !important; font-family: 'DM Sans', sans-serif !important; font-weight: 500 !important; font-size: 1rem !important; width: 100% !important; }
.stButton > button:hover { opacity: 0.85 !important; }
.sec-heading { font-family: 'Playfair Display', serif; font-size: 1.15rem; color: #4ade80; margin: 1.5rem 0 0.8rem; }
</style>
""", unsafe_allow_html=True)


# LOAD MODEL

@st.cache_resource
def load_model():
    model_path = "model.keras"
    file_id = "1zFKboV50eRwTI78yH_ThncR5uAoD01Gu"
    url = f'https://drive.google.com/uc?id={file_id}'
    
    if not os.path.exists(model_path):
        with st.spinner("Downloading Model from Drive..."):
            gdown.download(url, model_path, quiet=False)
    
    try:
        return tf.keras.models.load_model(model_path, compile=False)
    except Exception as e:
        st.error(f"Load Failure: {e}")
        return None

model = load_model()

if model is None:
    st.stop()

# 61 CLASS NAMES

class_names = [
    "APPLE - APPLE SCAB", "APPLE - BLACK ROT", "APPLE - CEDAR APPLE RUST", "APPLE - HEALTHY",
    "BLUEBERRY - HEALTHY", "CHERRY (INCLUDING SOUR) - POWDERY MILDEW", "CHERRY (INCLUDING SOUR) - HEALTHY",
    "CHERRY SPOT LEAF", "CORN (MAIZE) - CERCOSPORA LEAF SPOT GRAY LEAF SPOT", "CORN (MAIZE) - COMMON RUST",
    "CORN (MAIZE) - NORTHERN LEAF BLIGHT", "CORN (MAIZE) - HEALTHY", "GRAPE - BLACK ROT",
    "GRAPE - ESCA (BLACK MEASLES)", "GRAPE - LEAF BLIGHT (ISARIOPSIS LEAF SPOT)", "GRAPE - HEALTHY",
    "ORANGE - HUANGLONGBING (CITRUS GREENING)", "PEACH RUST LEAF", "PEACH - BACTERIAL SPOT",
    "PEACH - HEALTHY", "PEPPER, BELL - BACTERIAL SPOT", "PEPPER, BELL - HEALTHY",
    "POTATO MOSAIC VIRUS", "POTATO - EARLY BLIGHT", "POTATO - LATE BLIGHT", "POTATO - HEALTHY",
    "RASPBERRY LEAF SPOT", "RASPBERRY RUST LEAF", "RASPBERRY - HEALTHY", "RICE BACTERIAL LEAF BLIGHT",
    "RICE BROWN SPOT", "RICE LEAF BLAST", "RICE LEAF SCALD", "RICE NARROW BROWN SPOT",
    "RICE TUNGRO", "RICE HEALTHY", "SOYBEAN BACTERIAL BLIGHT", "SOYBEAN FROGEYE LEAF",
    "SOYBEAN RUST LEAF", "SOYBEAN SEPTORIA BROWN SPOT", "SOYBEAN - HEALTHY", "SQUASH - POWDERY MILDEW",
    "STRAWBERRY LEAF SPOT", "STRAWBERRY - LEAF SCORCH", "STRAWBERRY - HEALTHY", "TOMATO - BACTERIAL SPOT",
    "TOMATO - EARLY BLIGHT", "TOMATO - LATE BLIGHT", "TOMATO - LEAF MOLD", "TOMATO - SEPTORIA LEAF SPOT",
    "TOMATO - SPIDER MITES TWO-SPOTTED SPIDER MITE", "TOMATO - TARGET SPOT",
    "TOMATO - TOMATO YELLOW LEAF CURL VIRUS", "TOMATO - TOMATO MOSAIC VIRUS", "TOMATO - HEALTHY",
    "WHEAT BROWN RUST", "WHEAT LOOSE SMUT", "WHEAT SEPTORIA LEAF", "WHEAT YELLOW RUST",
    "WHEAT HEALTHY LEAF", "NOT A LEAF"
]


# DISEASE KNOWLEDGE BASE

disease_info = {
    "APPLE - APPLE SCAB": {"info": "Fungal disease causing dark scabby lesions on leaves and fruits.", "treatment": "Apply fungicides like captan or myclobutanil.", "prevention": "Remove fallen leaves, improve air circulation.", "advice": "Prune infected branches and monitor regularly."},
    "APPLE - BLACK ROT": {"info": "Fungal disease causing black rotting on fruits and leaves.", "treatment": "Use copper-based fungicide spray.", "prevention": "Remove mummified fruits and dead branches.", "advice": "Sanitize pruning tools between cuts."},
    "APPLE - CEDAR APPLE RUST": {"info": "Fungal disease causing orange spots on apple leaves.", "treatment": "Apply fungicides at bud break stage.", "prevention": "Remove nearby cedar trees if possible.", "advice": "Monitor during wet spring weather."},
    "APPLE - HEALTHY": {"info": "Healthy apple plant.", "treatment": "No treatment needed.", "prevention": "Maintain proper pruning and fertilization.", "advice": "Continue regular monitoring."},
    "BLUEBERRY - HEALTHY": {"info": "Healthy blueberry plant.", "treatment": "No treatment needed.", "prevention": "Maintain soil acidity and proper watering.", "advice": "Monitor for pests regularly."},
    "CHERRY (INCLUDING SOUR) - POWDERY MILDEW": {"info": "Fungal disease causing white powdery coating on leaves.", "treatment": "Use sulfur-based or potassium bicarbonate fungicide.", "prevention": "Ensure good air circulation around plants.", "advice": "Avoid overhead irrigation."},
    "CHERRY (INCLUDING SOUR) - HEALTHY": {"info": "Healthy cherry plant.", "treatment": "No treatment needed.", "prevention": "Regular pruning and monitoring.", "advice": "Check for pests seasonally."},
    "CHERRY SPOT LEAF": {"info": "Fungal disease causing reddish-brown spots on cherry leaves.", "treatment": "Apply copper fungicide or chlorothalonil.", "prevention": "Remove infected leaves promptly.", "advice": "Avoid wetting leaves during irrigation."},
    "CORN (MAIZE) - CERCOSPORA LEAF SPOT GRAY LEAF SPOT": {"info": "Fungal disease causing gray rectangular lesions on corn leaves.", "treatment": "Apply strobilurin fungicides.", "prevention": "Use resistant hybrids and practice crop rotation.", "advice": "Monitor fields during humid conditions."},
    "CORN (MAIZE) - COMMON RUST": {"info": "Fungal rust disease causing brown pustules on maize leaves.", "treatment": "Use fungicides like azoxystrobin.", "prevention": "Plant resistant seed varieties.", "advice": "Scout fields early in the season."},
    "CORN (MAIZE) - NORTHERN LEAF BLIGHT": {"info": "Fungal disease causing long tan lesions on corn leaves.", "treatment": "Apply propiconazole fungicide.", "prevention": "Use resistant hybrids and rotate crops.", "advice": "Remove crop debris after harvest."},
    "CORN (MAIZE) - HEALTHY": {"info": "Healthy maize plant.", "treatment": "No treatment needed.", "prevention": "Maintain proper spacing and fertilization.", "advice": "Continue monitoring for early signs."},
    "GRAPE - BLACK ROT": {"info": "Fungal disease causing black shriveled fruits and leaf spots.", "treatment": "Apply mancozeb or myclobutanil spray.", "prevention": "Prune and remove infected plant material.", "advice": "Spray preventively during wet weather."},
    "GRAPE - ESCA (BLACK MEASLES)": {"info": "Complex fungal disease causing tiger-stripe patterns on leaves.", "treatment": "No fully effective chemical treatment; remove infected wood.", "prevention": "Protect pruning wounds with fungicide paste.", "advice": "Avoid water stress in vines."},
    "GRAPE - LEAF BLIGHT (ISARIOPSIS LEAF SPOT)": {"info": "Fungal disease causing dark brown spots on grape leaves.", "treatment": "Use copper-based fungicides.", "prevention": "Remove infected leaves and improve airflow.", "advice": "Apply fungicide before rainy seasons."},
    "GRAPE - HEALTHY": {"info": "Healthy grape vine.", "treatment": "No treatment needed.", "prevention": "Regular pruning and canopy management.", "advice": "Monitor for powdery mildew in summer."},
    "ORANGE - HUANGLONGBING (CITRUS GREENING)": {"info": "Bacterial disease spread by insects causing yellowing and fruit drop.", "treatment": "No cure; remove infected trees to prevent spread.", "prevention": "Control psyllid insect vector with insecticides.", "advice": "Use certified disease-free planting material."},
    "PEACH RUST LEAF": {"info": "Fungal rust disease causing orange pustules on peach leaves.", "treatment": "Apply sulfur or copper fungicide.", "prevention": "Remove infected leaves and nearby alternate hosts.", "advice": "Monitor during warm humid conditions."},
    "PEACH - BACTERIAL SPOT": {"info": "Bacterial disease causing dark spots on leaves and fruits.", "treatment": "Apply copper-based bactericide sprays.", "prevention": "Avoid overhead irrigation, plant resistant varieties.", "advice": "Prune to improve air circulation."},
    "PEACH - HEALTHY": {"info": "Healthy peach plant.", "treatment": "No treatment needed.", "prevention": "Regular pruning and pest monitoring.", "advice": "Fertilize appropriately each season."},
    "PEPPER, BELL - BACTERIAL SPOT": {"info": "Bacterial disease causing water-soaked spots on leaves and fruits.", "treatment": "Use copper-based bactericide.", "prevention": "Use disease-free seeds and avoid leaf wetness.", "advice": "Rotate crops and remove infected plants."},
    "PEPPER, BELL - HEALTHY": {"info": "Healthy bell pepper plant.", "treatment": "No treatment needed.", "prevention": "Ensure proper watering and drainage.", "advice": "Monitor for aphids and spider mites."},
    "POTATO MOSAIC VIRUS": {"info": "Viral disease causing mosaic patterns and leaf distortion.", "treatment": "No cure; remove and destroy infected plants.", "prevention": "Use virus-free certified seed potatoes.", "advice": "Control aphid vectors with insecticide."},
    "POTATO - EARLY BLIGHT": {"info": "Fungal disease causing dark brown spots with concentric rings.", "treatment": "Apply chlorothalonil or mancozeb fungicide.", "prevention": "Crop rotation and proper plant spacing.", "advice": "Remove infected leaves promptly."},
    "POTATO - LATE BLIGHT": {"info": "Devastating fungal disease causing dark water-soaked lesions.", "treatment": "Use fungicides like metalaxyl immediately.", "prevention": "Avoid excess moisture and use resistant varieties.", "advice": "Destroy infected plants quickly to stop spread."},
    "POTATO - HEALTHY": {"info": "Healthy potato plant.", "treatment": "No treatment needed.", "prevention": "Use certified disease-free seed tubers.", "advice": "Monitor regularly for early symptoms."},
    "RASPBERRY LEAF SPOT": {"info": "Fungal disease causing circular spots on raspberry leaves.", "treatment": "Apply copper fungicide or captan.", "prevention": "Improve drainage and air circulation.", "advice": "Remove fallen infected leaves."},
    "RASPBERRY RUST LEAF": {"info": "Fungal rust causing yellow-orange pustules on leaves.", "treatment": "Apply sulfur or myclobutanil fungicide.", "prevention": "Remove infected canes and improve airflow.", "advice": "Avoid wetting foliage during watering."},
    "RASPBERRY - HEALTHY": {"info": "Healthy raspberry plant.", "treatment": "No treatment needed.", "prevention": "Annual pruning and cane management.", "advice": "Monitor for botrytis during fruiting."},
    "RICE BACTERIAL LEAF BLIGHT": {"info": "Bacterial disease causing water-soaked lesions on rice leaves.", "treatment": "Apply copper bactericide; use resistant varieties.", "prevention": "Avoid excess nitrogen fertilizer.", "advice": "Drain fields during severe outbreaks."},
    "RICE BROWN SPOT": {"info": "Fungal disease causing brown oval spots on rice leaves.", "treatment": "Use mancozeb or iprodione fungicide.", "prevention": "Balanced fertilization and proper water management.", "advice": "Use certified disease-free seeds."},
    "RICE LEAF BLAST": {"info": "Destructive fungal disease causing diamond-shaped lesions.", "treatment": "Apply tricyclazole or isoprothiolane fungicide.", "prevention": "Avoid excess nitrogen, use resistant varieties.", "advice": "Monitor during cool humid weather."},
    "RICE LEAF SCALD": {"info": "Fungal disease causing scalded appearance on rice leaves.", "treatment": "Apply propiconazole fungicide.", "prevention": "Balanced soil nutrition and good drainage.", "advice": "Remove infected plant debris after harvest."},
    "RICE NARROW BROWN SPOT": {"info": "Fungal disease causing narrow brown streaks on rice leaves.", "treatment": "Use mancozeb spray.", "prevention": "Proper fertilization and water management.", "advice": "Use resistant rice varieties."},
    "RICE TUNGRO": {"info": "Viral disease causing yellow-orange discoloration in rice.", "treatment": "No cure; remove infected plants immediately.", "prevention": "Control green leafhopper vector with insecticide.", "advice": "Plant tungro-resistant varieties."},
    "RICE HEALTHY": {"info": "Healthy rice plant.", "treatment": "No treatment needed.", "prevention": "Maintain proper water levels and fertilization.", "advice": "Scout regularly for leafhopper insects."},
    "SOYBEAN BACTERIAL BLIGHT": {"info": "Bacterial disease causing water-soaked angular leaf spots.", "treatment": "Apply copper-based bactericide.", "prevention": "Use disease-free seeds and avoid leaf wetness.", "advice": "Practice crop rotation with non-legumes."},
    "SOYBEAN FROGEYE LEAF": {"info": "Fungal disease causing circular gray spots with brown borders.", "treatment": "Apply azoxystrobin or pyraclostrobin fungicide.", "prevention": "Use resistant varieties and rotate crops.", "advice": "Scout fields during humid conditions."},
    "SOYBEAN RUST LEAF": {"info": "Aggressive fungal rust causing tan lesions and defoliation.", "treatment": "Apply triazole fungicides at first sign.", "prevention": "Early planting and use of resistant varieties.", "advice": "Monitor closely — spreads rapidly."},
    "SOYBEAN SEPTORIA BROWN SPOT": {"info": "Fungal disease causing brown irregular spots on lower leaves.", "treatment": "Use thiophanate-methyl fungicide.", "prevention": "Crop rotation and residue management.", "advice": "Apply fungicide at early pod stage."},
    "SOYBEAN - HEALTHY": {"info": "Healthy soybean plant.", "treatment": "No treatment needed.", "prevention": "Proper spacing and weed management.", "advice": "Monitor for aphids and bean beetles."},
    "SQUASH - POWDERY MILDEW": {"info": "Fungal disease causing white powdery coating on squash leaves.", "treatment": "Apply potassium bicarbonate or sulfur fungicide.", "prevention": "Ensure good air circulation and avoid overhead watering.", "advice": "Remove heavily infected leaves."},
    "STRAWBERRY LEAF SPOT": {"info": "Fungal disease causing purple-bordered spots on strawberry leaves.", "treatment": "Apply captan or copper fungicide.", "prevention": "Remove infected leaves and improve drainage.", "advice": "Avoid excessive nitrogen fertilizer."},
    "STRAWBERRY - LEAF SCORCH": {"info": "Fungal disease causing purple spots that dry and scorch leaves.", "treatment": "Use myclobutanil or copper fungicide.", "prevention": "Remove old leaves after harvest.", "advice": "Plant resistant varieties where available."},
    "STRAWBERRY - HEALTHY": {"info": "Healthy strawberry plant.", "treatment": "No treatment needed.", "prevention": "Regular renovation and weed control.", "advice": "Monitor for slugs and spider mites."},
    "TOMATO - BACTERIAL SPOT": {"info": "Bacterial disease causing small dark spots on leaves and fruits.", "treatment": "Use copper-based bactericide sprays.", "prevention": "Use disease-free seeds and avoid leaf wetness.", "advice": "Remove and destroy infected plant debris."},
    "TOMATO - EARLY BLIGHT": {"info": "Fungal disease causing brown spots with concentric rings.", "treatment": "Apply mancozeb or copper fungicide.", "prevention": "Avoid wet leaves and ensure plant spacing.", "advice": "Remove infected lower leaves immediately."},
    "TOMATO - LATE BLIGHT": {"info": "Highly destructive fungal disease causing dark water-soaked lesions.", "treatment": "Use chlorothalonil or metalaxyl fungicide.", "prevention": "Improve air circulation and avoid overhead watering.", "advice": "Destroy infected plants quickly to prevent spread."},
    "TOMATO - LEAF MOLD": {"info": "Fungal disease causing yellow patches and olive-green mold on leaves.", "treatment": "Apply chlorothalonil or mancozeb.", "prevention": "Reduce humidity, improve ventilation.", "advice": "Remove infected leaves immediately."},
    "TOMATO - SEPTORIA LEAF SPOT": {"info": "Fungal disease causing small circular spots with dark borders.", "treatment": "Apply copper fungicide or chlorothalonil.", "prevention": "Avoid splashing water on leaves.", "advice": "Stake plants to improve air circulation."},
    "TOMATO - SPIDER MITES TWO-SPOTTED SPIDER MITE": {"info": "Pest infestation causing yellow stippling and webbing on leaves.", "treatment": "Apply miticide or insecticidal soap.", "prevention": "Keep plants well-watered; avoid dry dusty conditions.", "advice": "Introduce natural predators like lacewings."},
    "TOMATO - TARGET SPOT": {"info": "Fungal disease causing concentric ring spots on leaves.", "treatment": "Apply azoxystrobin or chlorothalonil.", "prevention": "Improve plant spacing and air circulation.", "advice": "Remove infected plant debris promptly."},
    "TOMATO - TOMATO YELLOW LEAF CURL VIRUS": {"info": "Viral disease spread by whiteflies causing leaf curling and yellowing.", "treatment": "No cure; remove infected plants.", "prevention": "Control whitefly populations with insecticide.", "advice": "Use reflective mulches to repel whiteflies."},
    "TOMATO - TOMATO MOSAIC VIRUS": {"info": "Viral disease causing mosaic patterns and distorted leaves.", "treatment": "No cure; remove and destroy infected plants.", "prevention": "Use virus-free seeds and sanitize tools.", "advice": "Avoid smoking near plants — tobacco carries the virus."},
    "TOMATO - HEALTHY": {"info": "Healthy tomato plant.", "treatment": "No treatment needed.", "prevention": "Maintain proper watering and fertilization.", "advice": "Monitor regularly for early signs of disease."},
    "WHEAT BROWN RUST": {"info": "Fungal rust causing brown pustules on wheat leaves.", "treatment": "Apply propiconazole or tebuconazole fungicide.", "prevention": "Plant resistant varieties and monitor early.", "advice": "Apply fungicide at flag leaf stage."},
    "WHEAT LOOSE SMUT": {"info": "Fungal disease replacing grain with black smut spores.", "treatment": "Use systemic seed treatment fungicide.", "prevention": "Use certified smut-free seeds.", "advice": "Remove smutted heads before spore release."},
    "WHEAT SEPTORIA LEAF": {"info": "Fungal disease causing tan blotches on wheat leaves.", "treatment": "Apply azoxystrobin or propiconazole.", "prevention": "Rotate crops and use resistant varieties.", "advice": "Scout at tillering stage."},
    "WHEAT YELLOW RUST": {"info": "Fungal disease causing yellow stripe pustules on wheat leaves.", "treatment": "Apply propiconazole or mancozeb fungicide.", "prevention": "Plant resistant seed varieties.", "advice": "Monitor in cool moist conditions — spreads fast."},
    "WHEAT HEALTHY LEAF": {"info": "Healthy wheat plant.", "treatment": "No treatment needed.", "prevention": "Proper fertilization and weed control.", "advice": "Monitor for aphids and rust during growing season."},
    "NOT A LEAF": {"info": "This image does not appear to be a plant leaf.", "treatment": "No treatment needed.", "prevention": "Please upload a clear, close-up image of a plant leaf.", "advice": "Make sure the leaf fills most of the image and is well-lit."}
}

default_info = {
    "info": "Information not found.",
    "treatment": "Consult an agricultural expert.",
    "prevention": "Maintain crop hygiene.",
    "advice": "Monitor the plant regularly."
}

# PREDICTION LOGIC

def predict(image):

    img = tf.keras.preprocessing.image.load_img(image, target_size=(380, 380))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_preprocessed = preprocess_input(img_array)
    img_final = np.expand_dims(img_preprocessed, axis=0)
    
    preds = model.predict(img_final)
    top3_idx = np.argsort(preds[0])[::-1][:3]
    top3_probs = preds[0][top3_idx]
    return top3_idx, top3_probs

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



# UI RENDER

st.markdown("""
<div class="hero">
    <div class="hero-badge">AI-Powered · Final Year Project</div>
    <div class="hero-title">🌿 PlantAI</div>
    <div class="hero-sub">Smart Plant Disease Detection & Advisory System</div>
    <div class="stat-row">
        <div class="stat-pill"><strong>61</strong> Disease Classes</div>
        <div class="stat-pill"><strong>EfficientNet</strong> Architecture</div>
        <div class="stat-pill"><strong>380*380</strong> Resolution</div>
        <div class="stat-pill"><strong>3</strong> Languages</div>
    </div>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns([1, 1.2], gap="large")

with col1:
    st.markdown('<div class="sec-heading">🌍 Language</div>', unsafe_allow_html=True)
    language = st.selectbox("", ["English", "Nepali", "Hindi"], label_visibility="collapsed")

    st.markdown('<div class="sec-heading">📷 Upload Leaf Image</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"], label_visibility="collapsed")

    if uploaded_file:
        st.image(uploaded_file, use_container_width=True)
        predict_btn = st.button(" Analyze Disease")
    else:
        predict_btn = False

with col2:
    if uploaded_file and predict_btn:
        with st.spinner("🔬 Analyzing image..."):
            top3_idx, top3_probs = predict(uploaded_file)

        result = class_names[top3_idx[0]]
        confidence = top3_probs[0]

        if result == "NOT A LEAF":
            st.markdown(f"""
            <div class="error-box">
                <div class="error-title">Not a Plant Leaf</div>
                <div class="error-text">Upload a clear photo. (Confidence: {confidence*100:.1f}%)</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            is_healthy = "HEALTHY" in result
            color = "#4ade80" if is_healthy else "#f87171"
            
            st.markdown(f"""
            <div class="result-card" style="border-color:{color};">
                <div class="result-disease" style="color:{color};">{result}</div>
                <div class="result-conf">Confidence: {confidence * 100:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown('<div class="sec-heading">🏆 Top 3 Predictions</div>', unsafe_allow_html=True)
            for i in range(3):
                name = class_names[top3_idx[i]]
                pct = top3_probs[i] * 100
                st.markdown(f"""
                <div class="pred-bar-wrap">
                    <div class="pred-label">{name}</div>
                    <div class="pred-bar-bg"><div class="pred-bar-fill" style="width:{int(pct)}%"></div></div>
                    <div class="pred-pct">{pct:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)

            
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
            <div style="font-size:2.5rem;margin-bottom:1rem;"></div>
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
