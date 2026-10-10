import os
import requests
import numpy as np
import streamlit as st
import torch
import torch.nn as nn

from PIL import Image
from torchvision import models, transforms

# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="Military-Civilian Vehicle Intelligence",
    page_icon="🚙",
    layout="wide"
)

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

MODEL_URL = (
    "https://github.com/Devang-555/"
    "military-vs-civilian-vehicle-intelligence/"
    "releases/download/v1.0.0/"
    "efficientnet_b0_vehicle_champion.pth"
)

MODEL_FILE = "efficientnet_b0_vehicle_champion.pth"

DEFAULT_CLASSES = [
    "civilian aircraft",
    "civilian car",
    "military aircraft",
    "military helicopter",
    "military tank",
    "military truck",
]

IMAGE_SIZE = 224

# ---------------------------------------------------------
# Styling
# ---------------------------------------------------------

st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.4rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        color: #777;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }

    .result-box {
        padding: 1.2rem;
        border-radius: 12px;
        border: 1px solid rgba(128,128,128,0.25);
        margin-bottom: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------------------------------------------------
# Download model
# ---------------------------------------------------------

@st.cache_resource
def download_model():
    if not os.path.exists(MODEL_FILE):
        with st.spinner("Downloading champion model..."):
            response = requests.get(MODEL_URL, timeout=120)
            response.raise_for_status()

            with open(MODEL_FILE, "wb") as f:
                f.write(response.content)

    return MODEL_FILE


# ---------------------------------------------------------
# Load EfficientNet-B0
# ---------------------------------------------------------

@st.cache_resource
def load_model():

    model_path = download_model()

    checkpoint = torch.load(
        model_path,
        map_location="cpu",
        weights_only=False
    )

    model = models.efficientnet_b0(weights=None)

    num_features = model.classifier[1].in_features

    model.classifier[1] = nn.Linear(
        num_features,
        len(DEFAULT_CLASSES)
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()

    class_names = checkpoint.get(
        "class_names",
        DEFAULT_CLASSES
    )

    return model, class_names


# ---------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ---------------------------------------------------------
# Prediction
# ---------------------------------------------------------

def predict(image, model, class_names):

    image_rgb = image.convert("RGB")

    input_tensor = transform(image_rgb).unsqueeze(0)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]

    top_values, top_indices = torch.topk(
        probabilities,
        k=min(3, len(class_names))
    )

    results = []

    for probability, index in zip(
        top_values.tolist(),
        top_indices.tolist()
    ):
        results.append(
            (class_names[index], probability)
        )

    return results, probabilities


# ---------------------------------------------------------
# App
# ---------------------------------------------------------

st.markdown(
    '<div class="main-title">🚙 Military-Civilian Vehicle Intelligence</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'EfficientNet-B0 image classification demo'
    '</div>',
    unsafe_allow_html=True
)

st.info(
    "Educational computer-vision demonstration. "
    "This model is not validated for operational military, "
    "surveillance, targeting, or identification use."
)

uploaded_file = st.file_uploader(
    "Upload a vehicle image",
    type=["jpg", "jpeg", "png", "webp"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.image(
            image,
            caption="Uploaded image",
            use_container_width=True
        )

    with col2:

        model, class_names = load_model()

        with st.spinner("Analyzing image..."):

            results, probabilities = predict(
                image,
                model,
                class_names
            )

        top_class, top_confidence = results[0]

        military_indices = [
            i for i, name in enumerate(class_names)
            if name.startswith("military")
        ]

        civilian_indices = [
            i for i, name in enumerate(class_names)
            if name.startswith("civilian")
        ]

        military_probability = float(
            probabilities[military_indices].sum()
        )

        civilian_probability = float(
            probabilities[civilian_indices].sum()
        )

        if military_probability >= civilian_probability:
            group = "Military"
            group_confidence = military_probability
        else:
            group = "Civilian"
            group_confidence = civilian_probability

        st.markdown(
            '<div class="result-box">',
            unsafe_allow_html=True
        )

        st.subheader("🎯 Prediction")

        st.write(
            f"**Class:** {top_class.title()}"
        )

        st.metric(
            "Classification Confidence",
            f"{top_confidence * 100:.2f}%"
        )

        st.metric(
            "Vehicle Group",
            group
        )

        st.progress(group_confidence)

        st.write(
            f"**Military probability:** "
            f"{military_probability * 100:.2f}%"
        )

        st.write(
            f"**Civilian probability:** "
            f"{civilian_probability * 100:.2f}%"
        )

        st.markdown("</div>", unsafe_allow_html=True)

        st.subheader("📊 Top-3 Predictions")

        for rank, (name, probability) in enumerate(
            results,
            start=1
        ):
            st.write(
                f"**{rank}. {name.title()}** — "
                f"{probability * 100:.2f}%"
            )

            st.progress(float(probability))


# ---------------------------------------------------------
# Model information
# ---------------------------------------------------------

with st.expander("ℹ️ About the model"):

    st.write(
        "Champion model: EfficientNet-B0"
    )

    st.write(
        "**Test Accuracy:** 94.11%"
    )

    st.write(
        "**Test Macro-F1:** 93.76%"
    )

    st.write(
        "**Validation Macro-F1:** 95.49%"
    )

    st.write(
        "**Input:** 224 × 224 RGB image"
    )

    st.write(
        "**Classes:** 6"
    )

    st.write(
        "The trained checkpoint is downloaded from "
        "the project's GitHub v1.0.0 release."
)
