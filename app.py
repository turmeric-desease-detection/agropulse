import os
import warnings

# --- TensorFlow warnings कमी करण्यासाठी ---
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.filterwarnings("ignore")

import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image

# ------------------------------------------------------------------
# 🧠 CONFIG
# ------------------------------------------------------------------
INCEPTION_MODEL_PATH = "turmeric_inceptionv3_model.keras"

# तुझ्या dataset प्रमाणे 5 classes
CLASS_NAMES = [
    "Dry Leaf",
    "Healthy Leaf",
    "Leaf Blotch",
    "Rhizome Disease Root",
    "Rhizome Healthy Root",
]

# ------------------------------------------------------------------
# 🌟 STREAMLIT PAGE CONFIG
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Turmeric Plant Disease Detection",
    page_icon="🌱",
    layout="wide",
)


# ------------------------------------------------------------------
# 📦 MODEL LOADING
# ------------------------------------------------------------------
@st.cache_resource
def load_model():
    """Load InceptionV3 turmeric disease model."""
    if not os.path.exists(INCEPTION_MODEL_PATH):
        return None

    try:
        model = tf.keras.models.load_model(INCEPTION_MODEL_PATH, compile=False)
        # prediction साठी light compile
        model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            metrics=["accuracy"],
        )
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def preprocess_image(image: Image.Image) -> np.ndarray:
    """
    InceptionV3 साठी image preprocess:
    - RGB convert
    - 299x299 resize
    - batch dimension add
    NOTE: Model च्या आत preprocess_input आधीच आहे,
    म्हणून इथे /255 normalization करायचं नाही.
    """
    if image.mode != "RGB":
        image = image.convert("RGB")

    image = image.resize((299, 299))
    img_array = np.array(image).astype("float32")

    # ❌ हे काढलं:
    # img_array = img_array / 255.0

    # (1, 299, 299, 3)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array



def predict(model, image: Image.Image):
    """Single image prediction."""
    processed = preprocess_image(image)
    preds = model.predict(processed)
    probs = preds[0]
    predicted_index = int(np.argmax(probs))
    return predicted_index, probs


# ------------------------------------------------------------------
# 🌱 MAIN APP
# ------------------------------------------------------------------
def main():
    # ==== TOP HERO ====
    st.markdown(
        """
        <style>
        .title-main {
            font-size: 2.2rem;
            font-weight: 700;
            margin-bottom: 0;
        }
        .subtext-main {
            color: #6c757d;
            font-size: 0.95rem;
            margin-top: 0.1rem;
        }
        .upload-box {
            border: 2px dashed #4CAF50;
            padding: 1rem;
            border-radius: 0.8rem;
            background-color: #f8fff8;
        }
        .pill-ok {
            display:inline-block;
            padding: 0.2rem 0.7rem;
            border-radius: 999px;
            background-color:#d4edda;
            color:#155724;
            font-size:0.8rem;
            margin-left:0.4rem;
        }
        .pill-bad {
            display:inline-block;
            padding: 0.2rem 0.7rem;
            border-radius: 999px;
            background-color:#f8d7da;
            color:#721c24;
            font-size:0.8rem;
            margin-left:0.4rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    col_logo, col_title = st.columns([1, 4])
    with col_logo:
        st.markdown("### 🌿")
    with col_title:
        st.markdown("<div class='title-main'>Turmeric Plant Disease Detection</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='subtext-main'>Interactive AI app powered by InceptionV3 • Upload turmeric leaf and get instant disease prediction</div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ==== SIDEBAR ====
    with st.sidebar:
        st.header("ℹ️ App Info")
        model = load_model()
        if model is not None:
            st.success("Model loaded: **InceptionV3 turmeric disease model**")
        else:
            st.error("Model file not found!")
            st.info(
                f"""
**Note:**  
- Train your InceptionV3 model in notebook  
- Save it as: `{INCEPTION_MODEL_PATH}`  
- Keep it in same folder as `app.py`
"""
            )

        st.subheader("Detectable Classes")
        st.write(
            "- Dry Leaf\n"
            "- Healthy Leaf\n"
            "- Leaf Blotch\n"
            "- Rhizome Disease Root\n"
            "- Rhizome Healthy Root"
        )

        st.subheader("Tips for Best Results")
        st.write(
            """
- Use clear, close-up images of turmeric leaf  
- Avoid very dark / blurred photos  
- Avoid very small / zoomed-out plant images  
- Try 2–3 different angles if model gets confused
"""
        )

    # जर model load झाला नसेल तर पुढे काही करू नका
    model = load_model()
    if model is None:
        return

    # ==== TABS ====
    tab_predict, tab_how, tab_about = st.tabs(
        ["🔍 Predict Disease", "⚙️ How It Works", "📘 Model & Project Info"]
    )

    # ---------------------- PREDICT TAB ----------------------
    with tab_predict:
        view_mode = st.radio(
            "View mode",
            ["Simple", "Detailed"],
            horizontal=True,
            help="Simple: फक्त निकाल • Detailed: probabilities आणि माहिती",
        )

        col_left, col_right = st.columns([1.4, 1])

        # LEFT SIDE: Upload + Preview
        with col_left:
            st.subheader("Upload Image")

            st.markdown(
                "<div class='upload-box'><b>Drag & drop</b> turmeric leaf image here<br>"
                "<span style='font-size:0.85rem;color:#6c757d;'>Supported: JPG, JPEG, PNG</span></div>",
                unsafe_allow_html=True,
            )

            uploaded_file = st.file_uploader(
                "",
                type=["jpg", "jpeg", "png"],
                label_visibility="collapsed",
            )

            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_container_width=True)

        # RIGHT SIDE: Prediction
        with col_right:
            st.subheader("Prediction")

            if uploaded_file is None:
                st.info("👆 Drag & drop किंवा 'Browse files' वापरून turmeric leaf image upload करा.")
                disease_name = None
                probs = None
            else:
                with st.spinner("Analyzing leaf image with InceptionV3 model..."):
                    predicted_index, probs = predict(model, image)

                # index -> class नाव
                if predicted_index < len(CLASS_NAMES):
                    disease_name = CLASS_NAMES[predicted_index]
                else:
                    disease_name = f"Class {predicted_index}"

                # Result card
                if disease_name in ["Healthy Leaf", "Rhizome Healthy Root"]:
                    st.success(f"Prediction: **{disease_name}**  😄")
                    st.markdown("<span class='pill-ok'>Looks Healthy</span>", unsafe_allow_html=True)
                else:
                    st.error(f"Prediction: **{disease_name}**  ⚠️")
                    st.markdown("<span class='pill-bad'>Disease Detected</span>", unsafe_allow_html=True)

                # Simple / Detailed view
                if view_mode == "Detailed":
                    st.markdown("#### Class probabilities (model output)")

                    # length mismatch safe
                    if len(CLASS_NAMES) == len(probs):
                        classes_for_plot = CLASS_NAMES
                    else:
                        classes_for_plot = [f"Class {i}" for i in range(len(probs))]

                    probs_df = (
                        pd.DataFrame(
                            {
                                "Class": classes_for_plot,
                                "Probability": [float(p) for p in probs],
                            }
                        )
                        .set_index("Class")
                        .sort_values("Probability", ascending=False)
                    )

                    st.bar_chart(probs_df)

                    with st.expander("Show raw probabilities (numerical)", expanded=False):
                        st.dataframe(probs_df.style.format({"Probability": "{:.3f}"}))

        # Recommendations section
        if uploaded_file is not None:
            st.markdown("---")
            st.subheader("💡 Recommendations")

            if disease_name in ["Healthy Leaf", "Rhizome Healthy Root"]:
                st.success(
                    """
Your plant appears **healthy** ✅  

**Suggestions:**
- Maintain proper watering schedule  
- Ensure good soil and drainage  
- Monitor regularly for early signs of infection  
- Keep checking leaves weekly for any new spots or discoloration  
"""
                )
            elif disease_name == "Dry Leaf":
                st.warning(
                    """
**Dry Leaf Detected** 🟤  

**Possible reasons:**
- Water stress (over-watering / under-watering)  
- Nutrient deficiency  
- Too much direct sunlight / heat  

**Actions:**
- Check soil moisture (avoid fully dry or fully soggy)  
- Use balanced fertilizer as per crop guidelines  
- If pot-grown, move to partial shade during hottest hours  
"""
                )
            elif disease_name == "Leaf Blotch":
                st.warning(
                    """
**Leaf Blotch Detected** 🦠  

**Suggested actions:**
- Remove heavily affected leaves carefully  
- Use recommended fungicide (as per local agri expert)  
- Maintain good spacing between plants  
- Avoid overhead watering late evening (keeps leaves wet for long)  
"""
                )
            elif disease_name == "Rhizome Disease Root":
                st.error(
                    """
**Rhizome Disease Root Detected (Serious)** ⚠️  

**Immediate actions recommended:**
- Check soil drainage • Stop waterlogging immediately  
- Gently inspect rhizomes; remove & destroy badly infected parts  
- Consider soil treatment and crop rotation next season  
- Contact agricultural expert / Krishi Kendra for specific fungicide/advice  
"""
                )

    # ---------------------- HOW IT WORKS TAB ----------------------
    with tab_how:
        st.subheader("How the System Works")
        st.markdown(
            """
1. **Image Preprocessing**  
   - Image is resized to **299×299**  
   - Converted to RGB  
   - Normalized to range [0, 1]  

2. **Feature Extraction (InceptionV3)**  
   - Uses pre-trained **InceptionV3** (ImageNet)  
   - Only top layers retrained on turmeric leaf disease dataset  

3. **Prediction**  
   - Final Dense layer outputs probability for each class  
   - Class with highest probability is chosen as prediction  

4. **App Flow**  
   - User uploads image (drag & drop)  
   - Backend model runs on image  
   - App shows: predicted disease + detailed probabilities (optional)  
"""
        )

    # ---------------------- ABOUT TAB ----------------------
    with tab_about:
        st.subheader("Model & Project Info")
        st.markdown(
            """
- **Architecture:** InceptionV3 (transfer learning)  
- **Input Size:** 299×299 pixels, RGB  
- **Framework:** TensorFlow / Keras  
- **Frontend:** Streamlit  
- **Use Case:** Turmeric plant leaf disease detection  
"""
        )
        st.markdown("---")
        st.markdown(
            "Made with ❤️ to help farmers detect plant diseases early and protect their crops."
        )


if __name__ == "__main__":
    main()
