import os
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import streamlit as st
from PIL import Image
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Image as RLImage
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table

from model_loader import load_model, predict

try:
    from sklearn.ensemble import RandomForestRegressor

    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

st.set_page_config(page_title="RiceGenix AI", layout="centered")

rice_data = {
    "Gobindobhog": {"height": 51, "drought": 0, "disease": 1},
    "Swarna": {"height": 39, "drought": 0, "disease": 1},
    "Swarna Sub1": {"height": 39, "drought": 0, "disease": 1},
    "IR36": {"height": 35, "drought": 0, "disease": 1},
    "IR64": {"height": 39, "drought": 1, "disease": 1},
    "Minikit": {"height": 35, "drought": 0, "disease": 1},
    "Banskathi": {"height": 55, "drought": 1, "disease": 0},
    "Tulaipanji": {"height": 55, "drought": 0, "disease": 1},
    "Kalijeera": {"height": 50, "drought": 0, "disease": 1},
    "Radhatilak": {"height": 52, "drought": 0, "disease": 1},
    "Dudheswar": {"height": 50, "drought": 0, "disease": 1},
    "Kataribhog": {"height": 48, "drought": 0, "disease": 1},
    "Radhunipagal": {"height": 52, "drought": 0, "disease": 1},
    "Badshabhog": {"height": 50, "drought": 0, "disease": 1},
}

treatment_map = {
    "Bacterial Leaf Blight": "Apply Streptocycline + Copper oxychloride spray, use resistant varieties like Swarna.",
    "Brown Spot": "Apply Mancozeb fungicide, improve drainage, and add potassium nutrition.",
    "Leaf Blast": "Use Tricyclazole fungicide and avoid excess nitrogen fertilizer.",
    "Leaf Scald": "Use resistant varieties and apply a suitable systemic fungicide.",
    "Narrow Brown Leaf Spot": "Apply Carbendazim or Propiconazole and maintain field hygiene.",
    "Rice Hispa": "Apply Chlorpyrifos or Imidacloprid and remove affected leaves.",
    "Sheath Blight": "Apply Validamycin or Hexaconazole and reduce plant density.",
    "Healthy": "Crop looks healthy. Continue balanced irrigation and fertilization.",
    "AI Model Not Available": "The image model is not available right now, so disease detection could not run.",
    "Model Error": "The image model returned an error. Please retry with a clear crop image.",
    "Not Checked": "No crop image was uploaded, so disease detection was skipped.",
}


def estimate_ph(soil_type, water_source, fertilizer_use, manual_ph, ph_value):
    if manual_ph:
        return ph_value

    estimated_ph = 6.5

    if soil_type == "Clay":
        estimated_ph -= 0.5
    elif soil_type == "Sandy":
        estimated_ph += 0.3

    if water_source == "Groundwater":
        estimated_ph += 0.4

    if fertilizer_use == "Chemical":
        estimated_ph -= 0.6
    elif fertilizer_use == "Organic":
        estimated_ph += 0.2

    return estimated_ph


def crop_health(g2, rain, temp):
    disease = 1 if g2 == 0 else 0
    water = 1 if rain < 120 or temp > 42 else 0
    return disease, water


@st.cache_resource
def get_ai_model():
    return load_model()


@st.cache_resource
def get_yield_model():
    if not SKLEARN_AVAILABLE:
        return None

    np.random.seed(42)
    data_size = 300
    data = pd.DataFrame(
        {
            "Gene_A": np.random.randint(0, 2, data_size),
            "Gene_B": np.random.randint(0, 2, data_size),
            "Gene_C": np.random.randint(0, 2, data_size),
            "Gene_D": np.random.randint(0, 2, data_size),
            "Rain": np.random.randint(50, 500, data_size),
            "Temp": np.random.randint(20, 60, data_size),
            "pH": np.random.uniform(4.5, 8.5, data_size),
        }
    )

    data["Yield"] = (
        2
        + data["Gene_A"] * 0.9
        + data["Gene_B"] * 0.8
        + data["Gene_C"] * 0.7
        + data["Gene_D"] * 0.6
        + data["Rain"] * 0.012
        - (data["Temp"] - 30) * 0.06
        - abs(data["pH"] - 6.5) * 0.5
        + np.random.normal(0, 0.3, data_size)
    )

    features = data[["Gene_A", "Gene_B", "Gene_C", "Gene_D", "Rain", "Temp", "pH"]]
    target = data["Yield"]

    model = RandomForestRegressor(n_estimators=300, random_state=42)
    model.fit(features, target)
    return model


def generate_graphs(result):
    fig1, ax1 = plt.subplots()
    ax1.axvspan(0, 800, alpha=0.2)
    ax1.axvspan(800, 1500, alpha=0.2)
    ax1.axvspan(1500, 5000, alpha=0.2)

    rain_range = np.linspace(0, 5000, 100)
    ideal_yield = -((rain_range - 1200) ** 2) / 800000 + 6
    ax1.plot(rain_range, ideal_yield)
    ax1.scatter(result["rain"], result["yield"], s=100)
    ax1.set_xlabel("Rainfall (mm)")
    ax1.set_ylabel("Yield (tons/hectare)")
    ax1.set_title("Rainfall Impact on Yield")

    fig2, ax2 = plt.subplots()
    temp_range = np.linspace(0, 60, 100)
    ideal_yield_temp = -((temp_range - 30) ** 2) / 200 + 6
    ax2.plot(temp_range, ideal_yield_temp)
    ax2.scatter(result["temp"], result["yield"], s=100)
    ax2.set_xlabel("Temperature (C)")
    ax2.set_ylabel("Yield (tons/hectare)")
    ax2.set_title("Temperature Impact on Yield")

    return fig1, fig2


model_dl = get_ai_model()
yield_model = get_yield_model()

if "result" not in st.session_state:
    st.session_state.result = None

if st.sidebar.button("Reset Inputs"):
    st.session_state.clear()
    st.rerun()

st.sidebar.title("Settings")
theme = st.sidebar.selectbox("Theme", ["Dark", "Light"])
font_size = st.sidebar.slider("Font Size", 12, 26, 16)

st.sidebar.markdown("---")
st.sidebar.subheader("Support")
st.sidebar.write("Contact: 9832114844")
st.sidebar.write("Email: tamal.bot@gmail.com")

bg = "#0E1117" if theme == "Dark" else "white"
text = "white" if theme == "Dark" else "black"
st.markdown(
    f"""
    <style>
    .stApp {{ background-color: {bg}; color: {text}; }}
    html, body, [class*="css"] {{ font-size: {font_size}px; color: {text}; }}
    .bottom-space {{ height: 80px; }}
    </style>
    <div class="bottom-space"></div>
    """,
    unsafe_allow_html=True,
)

logo_path = "assets/logo.png"
col1, col2 = st.columns([1, 3])

with col1:
    if os.path.exists(logo_path):
        st.image(logo_path, width=120)

with col2:
    st.markdown("<h1>RiceGenix AI</h1>", unsafe_allow_html=True)

st.sidebar.subheader("Live Weather")
try:
    url = "https://api.open-meteo.com/v1/forecast?latitude=23.23&longitude=87.07&current_weather=true"
    weather = requests.get(url, timeout=10).json()
    temp_live = weather["current_weather"]["temperature"]
    st.sidebar.write(f"Temperature: {temp_live} C")
except Exception:
    st.sidebar.write("Weather unavailable")

if model_dl is None:
    st.info("AI disease model is unavailable, but yield prediction can still run.")

if not SKLEARN_AVAILABLE or yield_model is None:
    st.error("Prediction model failed to load. Please check deployment.")
    st.stop()

st.subheader("Plant Characteristics and Soil Details")

with st.form("prediction_form"):
    crop_name = st.selectbox("Select Rice Variety", list(rice_data.keys()))
    height = st.number_input("Enter plant height (in inches)", min_value=0.0, value=0.0)
    gene_b = st.radio("Disease resistant?", ["Yes", "No"], horizontal=True)
    gene_c = st.radio("Drought tolerant?", ["Yes", "No"], horizontal=True)
    gene_d = st.radio("Fast growth?", ["Yes", "No"], horizontal=True)
    rain = st.number_input("Annual Rainfall (mm)", min_value=0.0, value=0.0)
    temp_unit = st.selectbox("Temperature Unit", ["Celsius", "Fahrenheit"])
    temp_input = st.number_input("Average Temperature", value=0.0)

    st.markdown("### Soil Conditions")
    soil_type = st.selectbox("Soil Type", ["Loamy", "Clay", "Sandy", "Alluvial", "Laterite"])
    water_source = st.selectbox("Irrigation Water Type", ["Rainwater", "Groundwater", "Mixed"])
    fertilizer_use = st.selectbox("Fertilizer Usage", ["Organic", "Chemical", "Mixed"])
    manual_ph = st.checkbox("I know my soil pH")
    ph_input = st.slider("Enter Soil pH", 3.0, 9.0, 6.5, step=0.1, disabled=not manual_ph)

    uploaded_file = st.file_uploader(
        "Upload rice crop image (optional)",
        type=["jpg", "jpeg", "png"],
    )
    submitted = st.form_submit_button("Predict Yield", use_container_width=True)

if uploaded_file:
    preview_image = Image.open(uploaded_file).convert("RGB")
    st.image(preview_image, caption="Uploaded Image", use_container_width=True)
    uploaded_file.seek(0)

if submitted:
    st.session_state.result = None

    try:
        rain_val = float(rain)
        temp_val = float(temp_input)
        height_val = float(height)
    except Exception:
        st.error("Please enter valid numeric values for rainfall, temperature, and height.")
        st.stop()

    if temp_unit == "Fahrenheit":
        temp_val = (temp_val - 32) * 5 / 9

    ph = estimate_ph(soil_type, water_source, fertilizer_use, manual_ph, ph_input)
    g2 = 1 if gene_b == "Yes" else 0
    g3 = 1 if gene_c == "Yes" else 0
    g4 = 1 if gene_d == "Yes" else 0
    expected_height = rice_data[crop_name]["height"]
    g1 = 1 if height_val >= expected_height else 0

    disease_name = "Not Checked"
    image_for_report = None
    if uploaded_file:
        try:
            uploaded_file.seek(0)
            image_for_prediction = Image.open(uploaded_file).convert("RGB")
            disease_name = predict(model_dl, image_for_prediction)
            image_for_report = image_for_prediction.copy()
        except Exception:
            disease_name = "Model Error"

    base_pred = yield_model.predict([[g1, g2, g3, g4, rain_val, temp_val, ph]])[0]
    disease, water = crop_health(g2, rain_val, temp_val)
    final_pred = float(base_pred)

    if height_val < expected_height:
        final_pred -= 0.7
        height_flag = "Short growth detected -> yield may decrease."
    elif height_val > expected_height + 10:
        final_pred -= 0.3
        height_flag = "Overgrowth detected -> possible nutrient imbalance."
    else:
        height_flag = "Normal growth."

    if disease:
        final_pred -= 0.8

    if disease_name not in {"Healthy", "Not Checked", "AI Model Not Available"}:
        final_pred -= 1.2

    if rice_data[crop_name]["disease"] == 1 and g2 == 0:
        final_pred -= 0.5

    if water:
        final_pred -= 0.6

    if ph < 5.5:
        final_pred -= 0.5
    elif ph > 7.5:
        final_pred -= 0.4

    uploaded_image_path = None
    if image_for_report is not None:
        temp_buffer = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        image_for_report.save(temp_buffer.name)
        temp_buffer.close()
        uploaded_image_path = temp_buffer.name

    st.session_state.result = {
        "yield": final_pred,
        "genes": [g1, g2, g3, g4],
        "rain": rain_val,
        "temp": temp_val,
        "ph": ph,
        "disease": disease,
        "water": water,
        "disease_name": disease_name,
        "treatment": treatment_map.get(disease_name, "No treatment available."),
        "crop_name": crop_name,
        "height_flag": height_flag,
        "input_height": height_val,
        "expected_height": expected_height,
        "uploaded_image_path": uploaded_image_path,
    }

if st.session_state.result:
    res = st.session_state.result
    disease_val = res.get("disease", 0)
    water_val = res.get("water", 0)

    st.success(f"Predicted Yield: {res['yield']:.2f} tons/hectare")
    st.write(f"Soil pH: {res['ph']:.2f}")
    st.write(f"Crop Selected: {res['crop_name']}")
    st.write(f"Expected Height: {res['expected_height']} inches")
    st.write(f"Your Crop Height: {res['input_height']} inches")
    st.write(res["height_flag"])

    st.subheader("Gene Representation")
    st.write(
        {
            "Gene_A": res["genes"][0],
            "Gene_B": res["genes"][1],
            "Gene_C": res["genes"][2],
            "Gene_D": res["genes"][3],
        }
    )

    st.subheader("Crop Health Analysis")
    st.write(f"Disease Risk: {'High' if disease_val else 'Low'}")
    st.write(f"Water Stress: {'Yes' if water_val else 'No'}")
    st.write(f"Detected Disease: {res['disease_name']}")

    st.subheader("AI Suggestions")
    if res["disease_name"] not in {"Healthy", "Not Checked"}:
        st.write(f"Treatment: {res['treatment']}")
        st.write("Control disease early to prevent major yield loss.")
    elif res["disease_name"] == "Healthy":
        st.write(res["treatment"])

    if res["input_height"] < res["expected_height"]:
        st.write("Plant growth is below expected -> improve nitrogen supply and irrigation.")
    elif res["input_height"] > res["expected_height"] + 10:
        st.write("Excessive growth -> reduce nitrogen and balance nutrients.")

    if res["ph"] < 5.5:
        st.write("Soil is acidic -> add lime or dolomite to increase pH.")
        st.write("Use organic compost to stabilize soil condition.")
    elif res["ph"] > 7.5:
        st.write("Soil is alkaline -> add gypsum or organic matter.")
        st.write("Avoid excessive chemical fertilizers.")

    if res["rain"] < 100:
        st.write("Very low rainfall -> increase irrigation frequency.")
    elif res["rain"] < 200:
        st.write("Moderately low rainfall -> maintain regular irrigation.")
    elif res["rain"] > 400:
        st.write("Excess rainfall -> improve drainage system.")

    if res["temp"] > 40:
        st.write("High temperature stress -> maintain water level in the field.")
    elif res["temp"] < 20:
        st.write("Low temperature may slow growth, so monitor the crop carefully.")

    st.write("Use high-yield variety seeds like Swarna or IR64.")
    st.write("Apply balanced NPK fertilizers based on soil testing.")
    st.write("Monitor the crop weekly for early disease detection.")
    st.write("Use proper spacing to avoid fungal infection.")

    st.subheader("Data Analysis")
    fig1, fig2 = generate_graphs(res)
    st.pyplot(fig1)
    st.pyplot(fig2)

    if st.button("Export Report as PDF"):
        pdf_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        doc = SimpleDocTemplate(pdf_file.name, pagesize=A4)
        styles = getSampleStyleSheet()
        content = []

        if os.path.exists(logo_path):
            logo_img = RLImage(logo_path, width=80, height=80)
            title = Paragraph("<b><font size=18>RiceGenix AI Report</font></b>", styles["Title"])
            content.append(Table([[logo_img, title]]))
        else:
            content.append(Paragraph("<b><font size=18>RiceGenix AI Report</font></b>", styles["Title"]))

        content.append(Spacer(1, 12))
        content.append(Paragraph(f"<b>Crop Name:</b> {res['crop_name']}", styles["Normal"]))
        content.append(Paragraph(f"<b>Predicted Yield:</b> {res['yield']:.2f} tons/hectare", styles["Normal"]))
        content.append(Paragraph(f"<b>Rainfall:</b> {res['rain']} mm", styles["Normal"]))
        content.append(Paragraph(f"<b>Temperature:</b> {res['temp']:.2f} C", styles["Normal"]))
        content.append(Paragraph(f"<b>Soil pH:</b> {res['ph']:.2f}", styles["Normal"]))
        content.append(Spacer(1, 10))
        content.append(Paragraph("<b>Crop Health Analysis</b>", styles["Heading2"]))
        content.append(Paragraph(f"Disease Risk: {'High' if disease_val else 'Low'}", styles["Normal"]))
        content.append(Paragraph(f"Water Stress: {'Yes' if water_val else 'No'}", styles["Normal"]))
        content.append(Paragraph(f"Detected Disease: {res['disease_name']}", styles["Normal"]))
        content.append(Paragraph(f"<b>Treatment:</b> {res['treatment']}", styles["Normal"]))
        content.append(Spacer(1, 10))

        temp_graph1 = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        temp_graph2 = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        fig1.savefig(temp_graph1.name)
        fig2.savefig(temp_graph2.name)
        temp_graph1.close()
        temp_graph2.close()

        content.append(Paragraph("<b>Graphs</b>", styles["Heading2"]))
        content.append(RLImage(temp_graph1.name, width=400, height=250))
        content.append(Spacer(1, 10))
        content.append(RLImage(temp_graph2.name, width=400, height=250))

        if res.get("uploaded_image_path"):
            content.append(Spacer(1, 10))
            content.append(Paragraph("<b>Uploaded Crop Image</b>", styles["Heading2"]))
            content.append(RLImage(res["uploaded_image_path"], width=250, height=180))

        content.append(Spacer(1, 20))
        content.append(Paragraph("<i>A Project by Aranyak</i>", styles["Italic"]))
        doc.build(content)

        with open(pdf_file.name, "rb") as pdf_stream:
            st.download_button(
                label="Download PDF",
                data=pdf_stream.read(),
                file_name="RiceGenix_Report.pdf",
                mime="application/pdf",
                key="pdf_download",
            )
