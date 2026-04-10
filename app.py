import os
import tempfile
import base64
import textwrap

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


def rgba_to_hex(color_str):
    """Convert rgba() CSS string to hex format for matplotlib compatibility."""
    if not isinstance(color_str, str):
        return "#102235"  # fallback color
    
    color_str = color_str.strip()
    
    # If already a hex color, return as-is
    if color_str.startswith("#"):
        return color_str
    
    # Convert rgba() format to hex
    if color_str.startswith("rgba"):
        try:
            # Extract numbers from rgba(r, g, b, a)
            numbers = color_str.replace("rgba(", "").replace(")", "").split(",")
            r, g, b = int(numbers[0].strip()), int(numbers[1].strip()), int(numbers[2].strip())
            return f"#{r:02x}{g:02x}{b:02x}"
        except (ValueError, IndexError):
            return "#102235"  # fallback color
    
    # Return as-is if it's a named color or valid format
    return color_str

st.set_page_config(page_title="RiceGenixAI", layout="wide", initial_sidebar_state="collapsed")

rice_data = {
    "Gobindobhog": {"height": 51, "drought": 0, "disease": 1, "maturity_months": 4.5},
    "Swarna": {"height": 39, "drought": 0, "disease": 1, "maturity_months": 4.0},
    "Swarna Sub1": {"height": 39, "drought": 0, "disease": 1, "maturity_months": 4.0},
    "IR36": {"height": 35, "drought": 0, "disease": 1, "maturity_months": 3.8},
    "IR64": {"height": 39, "drought": 1, "disease": 1, "maturity_months": 4.0},
    "Minikit": {"height": 35, "drought": 0, "disease": 1, "maturity_months": 3.8},
    "Banskathi": {"height": 55, "drought": 1, "disease": 0, "maturity_months": 5.0},
    "Tulaipanji": {"height": 55, "drought": 0, "disease": 1, "maturity_months": 5.0},
    "Kalijeera": {"height": 50, "drought": 0, "disease": 1, "maturity_months": 4.8},
    "Radhatilak": {"height": 52, "drought": 0, "disease": 1, "maturity_months": 4.8},
    "Dudheswar": {"height": 50, "drought": 0, "disease": 1, "maturity_months": 4.8},
    "Kataribhog": {"height": 48, "drought": 0, "disease": 1, "maturity_months": 4.8},
    "Radhunipagal": {"height": 52, "drought": 0, "disease": 1, "maturity_months": 4.8},
    "Badshabhog": {"height": 50, "drought": 0, "disease": 1, "maturity_months": 4.8},
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


def project_growth_metrics(crop_name, actual_height, months_observed):
    crop_info = rice_data[crop_name]
    expected_final_height = float(crop_info["height"])
    maturity_months = float(crop_info["maturity_months"])
    normalized_months = max(0.5, min(float(months_observed), maturity_months))
    growth_ratio = min(normalized_months / maturity_months, 1.0)
    expected_height_now = expected_final_height * growth_ratio

    raw_projected_height = actual_height / growth_ratio
    projected_final_height = max(
        actual_height,
        min(expected_final_height * 1.35, raw_projected_height * 0.6 + expected_final_height * 0.4),
    )

    return {
        "maturity_months": maturity_months,
        "expected_height_now": expected_height_now,
        "projected_final_height": projected_final_height,
        "growth_ratio": growth_ratio,
    }


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
    theme_tokens = result.get(
        "theme_tokens",
        {
            "bg": "#08131f",
            "surface": "#102235",
            "text": "#eaf4ff",
            "muted": "#9db5c7",
            "accent": "#30c48d",
            "accent_soft": "#76e4bc",
        },
    )
    
    # Sanitize colors for matplotlib compatibility
    theme_tokens = {
        k: rgba_to_hex(v) for k, v in theme_tokens.items()
    }

    plt.style.use("default")
    fig1, ax1 = plt.subplots()
    fig1.patch.set_facecolor(theme_tokens["surface"])
    ax1.set_facecolor(theme_tokens["bg"])
    ax1.axvspan(0, 800, alpha=0.2)
    ax1.axvspan(800, 1500, alpha=0.2)
    ax1.axvspan(1500, 5000, alpha=0.2)

    rain_range = np.linspace(0, 5000, 100)
    ideal_yield = -((rain_range - 1200) ** 2) / 800000 + 6
    ax1.plot(rain_range, ideal_yield, color=theme_tokens["accent"], linewidth=3)
    ax1.scatter(result["rain"], result["yield"], s=110, color=theme_tokens["accent_soft"], edgecolors="white", linewidths=1.4)
    ax1.set_xlabel("Rainfall (mm)")
    ax1.set_ylabel("Yield (tons/hectare)")
    ax1.set_title("Rainfall Impact on Yield")
    ax1.tick_params(colors=theme_tokens["muted"])
    ax1.xaxis.label.set_color(theme_tokens["text"])
    ax1.yaxis.label.set_color(theme_tokens["text"])
    ax1.title.set_color(theme_tokens["text"])
    for spine in ax1.spines.values():
        spine.set_color(theme_tokens["muted"])

    fig2, ax2 = plt.subplots()
    fig2.patch.set_facecolor(theme_tokens["surface"])
    ax2.set_facecolor(theme_tokens["bg"])
    temp_range = np.linspace(0, 60, 100)
    ideal_yield_temp = -((temp_range - 30) ** 2) / 200 + 6
    ax2.plot(temp_range, ideal_yield_temp, color=theme_tokens["accent"], linewidth=3)
    ax2.scatter(result["temp"], result["yield"], s=110, color=theme_tokens["accent_soft"], edgecolors="white", linewidths=1.4)
    ax2.set_xlabel("Temperature (C)")
    ax2.set_ylabel("Yield (tons/hectare)")
    ax2.set_title("Temperature Impact on Yield")
    ax2.tick_params(colors=theme_tokens["muted"])
    ax2.xaxis.label.set_color(theme_tokens["text"])
    ax2.yaxis.label.set_color(theme_tokens["text"])
    ax2.title.set_color(theme_tokens["text"])
    for spine in ax2.spines.values():
        spine.set_color(theme_tokens["muted"])

    return fig1, fig2


def build_pdf_report(result, fig1, fig2, logo_path):
    disease_val = result.get("disease", 0)
    water_val = result.get("water", 0)
    pdf_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    doc = SimpleDocTemplate(pdf_file.name, pagesize=A4)
    styles = getSampleStyleSheet()
    content = []

    if os.path.exists(logo_path):
        logo_img = RLImage(logo_path, width=80, height=80)
        title = Paragraph("<b><font size=18>RiceGenixAI Report</font></b>", styles["Title"])
        content.append(Table([[logo_img, title]]))
    else:
        content.append(Paragraph("<b><font size=18>RiceGenixAI Report</font></b>", styles["Title"]))

    content.append(Spacer(1, 12))
    content.append(Paragraph(f"<b>Crop Name:</b> {result['crop_name']}", styles["Normal"]))
    content.append(Paragraph(f"<b>Predicted Yield:</b> {result['yield']:.2f} tons/hectare", styles["Normal"]))
    content.append(Paragraph(f"<b>Rainfall:</b> {result['rain']} mm", styles["Normal"]))
    content.append(Paragraph(f"<b>Temperature:</b> {result['temp']:.2f} C", styles["Normal"]))
    content.append(Paragraph(f"<b>Soil pH:</b> {result['ph']:.2f}", styles["Normal"]))
    content.append(Paragraph(f"<b>Months Observed:</b> {result['months_observed']:.1f}", styles["Normal"]))
    content.append(Paragraph(f"<b>Expected Height:</b> {result['expected_height']} inches", styles["Normal"]))
    content.append(Paragraph(f"<b>Expected Height Now:</b> {result['expected_height_now']:.2f} inches", styles["Normal"]))
    content.append(Paragraph(f"<b>Your Crop Height:</b> {result['input_height']} inches", styles["Normal"]))
    content.append(Paragraph(f"<b>Predicted Final Height:</b> {result['projected_final_height']:.2f} inches", styles["Normal"]))
    content.append(Paragraph(f"<b>Height Status:</b> {result['height_flag']}", styles["Normal"]))
    content.append(Spacer(1, 10))
    content.append(Paragraph("<b>Gene Representation</b>", styles["Heading2"]))
    for index, gene_name in enumerate(["Gene_A", "Gene_B", "Gene_C", "Gene_D"]):
        content.append(Paragraph(f"{gene_name}: {result['genes'][index]}", styles["Normal"]))

    content.append(Spacer(1, 10))
    content.append(Paragraph("<b>Crop Health Analysis</b>", styles["Heading2"]))
    content.append(Paragraph(f"Disease Risk: {'High' if disease_val else 'Low'}", styles["Normal"]))
    content.append(Paragraph(f"Water Stress: {'Yes' if water_val else 'No'}", styles["Normal"]))
    content.append(Paragraph(f"Detected Disease: {result['disease_name']}", styles["Normal"]))
    content.append(Paragraph(f"<b>Treatment:</b> {result['treatment']}", styles["Normal"]))

    temp_graph1 = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    temp_graph2 = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    fig1.savefig(temp_graph1.name)
    fig2.savefig(temp_graph2.name)
    temp_graph1.close()
    temp_graph2.close()

    content.append(Spacer(1, 10))
    content.append(Paragraph("<b>Graphs</b>", styles["Heading2"]))
    content.append(RLImage(temp_graph1.name, width=400, height=250))
    content.append(Spacer(1, 10))
    content.append(RLImage(temp_graph2.name, width=400, height=250))

    if result.get("uploaded_image_path"):
        content.append(Spacer(1, 10))
        content.append(Paragraph("<b>Uploaded Crop Image</b>", styles["Heading2"]))
        content.append(RLImage(result["uploaded_image_path"], width=250, height=180))

    content.append(Spacer(1, 20))
    content.append(Paragraph("<i>A Project by Aranyak Chakraborty</i>", styles["Italic"]))
    doc.build(content)

    with open(pdf_file.name, "rb") as pdf_stream:
        return pdf_stream.read()


def build_input_signature(crop_name, height, months_observed, gene_b, gene_c, rain, temp_unit, temp_input, soil_type, water_source, fertilizer_use, manual_ph, ph_input, has_image):
    return (
        crop_name,
        float(height),
        float(months_observed),
        gene_b,
        gene_c,
        float(rain),
        temp_unit,
        float(temp_input),
        soil_type,
        water_source,
        fertilizer_use,
        bool(manual_ph),
        float(ph_input),
        bool(has_image),
    )


def resolve_image_input(uploaded_file, camera_file):
    image_source = camera_file or uploaded_file
    if image_source is None:
        return None, None

    try:
        image_source.seek(0)
    except Exception:
        pass

    image = Image.open(image_source).convert("RGB")
    return image, image_source


@st.cache_data
def load_logo_data_uri(path):
    if not os.path.exists(path):
        return ""
    with open(path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode()
    return f"data:image/png;base64,{encoded}"


def default_ui_settings():
    return {
        "theme_mode": "System Default",
        "accent_palette": "Aurora",
        "font_family": "Manrope",
        "font_scale": 16,
        "radius": 24,
        "surface_style": "Glass",
        "motion": "Dynamic",
        "layout_density": "Comfortable",
        "hero_glow": True,
    }


def get_theme_tokens(theme_mode, accent_palette):
    palette_map = {
        "Aurora": {"accent": "#4ade80", "accent_soft": "#99f6c4", "accent_alt": "#22d3ee"},
        "Sunset": {"accent": "#fb7185", "accent_soft": "#fda4af", "accent_alt": "#f59e0b"},
        "Ocean": {"accent": "#38bdf8", "accent_soft": "#7dd3fc", "accent_alt": "#2dd4bf"},
        "Royal": {"accent": "#a78bfa", "accent_soft": "#c4b5fd", "accent_alt": "#f472b6"},
    }
    accent_tokens = palette_map.get(accent_palette, palette_map["Aurora"])

    dark = {
        "bg": "#07111c",
        "bg_alt": "#0d1b2b",
        "surface": "rgba(12, 24, 39, 0.78)",
        "surface_solid": "#102235",
        "card_border": "rgba(255, 255, 255, 0.1)",
        "text": "#f3f7fb",
        "muted": "#9fb3c6",
        "shadow": "0 24px 80px rgba(2, 10, 20, 0.45)",
        **accent_tokens,
    }
    light = {
        "bg": "#f4f8fc",
        "bg_alt": "#ffffff",
        "surface": "rgba(255, 255, 255, 0.82)",
        "surface_solid": "#ffffff",
        "card_border": "rgba(13, 26, 41, 0.08)",
        "text": "#0d1a29",
        "muted": "#627589",
        "shadow": "0 18px 60px rgba(79, 111, 143, 0.14)",
        **accent_tokens,
    }
    active = dark if theme_mode == "Dark" else light
    return {"dark": dark, "light": light, "active": active}


def inject_custom_theme(settings, token_pack):
    compact_gap = "0.8rem" if settings["layout_density"] == "Compact" else "1.15rem"
    motion_seconds = "0.45s" if settings["motion"] == "Dynamic" else "0.18s"
    glass_alpha = "rgba(255,255,255,0.08)" if settings["surface_style"] == "Glass" else "transparent"
    font_map = {
        "Manrope": "'Manrope', sans-serif",
        "Sora": "'Sora', sans-serif",
        "Poppins": "'Poppins', sans-serif",
    }
    font_family = font_map.get(settings["font_family"], "'Manrope', sans-serif")
    radius = settings["radius"]
    glow_opacity = "1" if settings["hero_glow"] else "0"

    if settings["theme_mode"] == "System Default":
        token_css = f"""
        :root {{
            --rg-bg: {token_pack['light']['bg']};
            --rg-bg-alt: {token_pack['light']['bg_alt']};
            --rg-surface: {token_pack['light']['surface']};
            --rg-surface-solid: {token_pack['light']['surface_solid']};
            --rg-text: {token_pack['light']['text']};
            --rg-muted: {token_pack['light']['muted']};
            --rg-border: {token_pack['light']['card_border']};
            --rg-shadow: {token_pack['light']['shadow']};
            --rg-accent: {token_pack['light']['accent']};
            --rg-accent-soft: {token_pack['light']['accent_soft']};
            --rg-accent-alt: {token_pack['light']['accent_alt']};
        }}
        @media (prefers-color-scheme: dark) {{
            :root {{
                --rg-bg: {token_pack['dark']['bg']};
                --rg-bg-alt: {token_pack['dark']['bg_alt']};
                --rg-surface: {token_pack['dark']['surface']};
                --rg-surface-solid: {token_pack['dark']['surface_solid']};
                --rg-text: {token_pack['dark']['text']};
                --rg-muted: {token_pack['dark']['muted']};
                --rg-border: {token_pack['dark']['card_border']};
                --rg-shadow: {token_pack['dark']['shadow']};
                --rg-accent: {token_pack['dark']['accent']};
                --rg-accent-soft: {token_pack['dark']['accent_soft']};
                --rg-accent-alt: {token_pack['dark']['accent_alt']};
            }}
        }}
        """
    else:
        active = token_pack["active"]
        token_css = f"""
        :root {{
            --rg-bg: {active['bg']};
            --rg-bg-alt: {active['bg_alt']};
            --rg-surface: {active['surface']};
            --rg-surface-solid: {active['surface_solid']};
            --rg-text: {active['text']};
            --rg-muted: {active['muted']};
            --rg-border: {active['card_border']};
            --rg-shadow: {active['shadow']};
            --rg-accent: {active['accent']};
            --rg-accent-soft: {active['accent_soft']};
            --rg-accent-alt: {active['accent_alt']};
        }}
        """

    st.markdown(
        textwrap.dedent(
            f"""
            <style>
            @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&family=Poppins:wght@400;500;600;700&family=Sora:wght@400;500;600;700&display=swap');
            {token_css}

            html, body, [class*="css"], .stApp {{
                font-family: {font_family};
                font-size: {settings['font_scale']}px;
                color: var(--rg-text);
            }}

            .stApp {{
                background:
                    radial-gradient(circle at 12% 18%, color-mix(in srgb, var(--rg-accent) 24%, transparent) 0%, transparent 34%),
                    radial-gradient(circle at 88% 14%, color-mix(in srgb, var(--rg-accent-alt) 24%, transparent) 0%, transparent 30%),
                    linear-gradient(160deg, var(--rg-bg) 0%, var(--rg-bg-alt) 100%);
                color: var(--rg-text);
            }}

            [data-testid="collapsedControl"], [data-testid="stSidebar"], #MainMenu, footer, header {{
                display: none !important;
            }}

            .block-container {{
                padding-top: 2rem;
                padding-bottom: 2.5rem;
                max-width: 1400px;
            }}

            .rg-shell {{
                position: relative;
                animation: rg-fade-up {motion_seconds} ease-out;
            }}

            .rg-hero {{
                position: relative;
                overflow: hidden;
                border-radius: {radius + 10}px;
                padding: 1.4rem 1.5rem;
                background: linear-gradient(140deg, var(--rg-surface) 0%, color-mix(in srgb, var(--rg-surface-solid) 72%, transparent) 100%);
                border: 1px solid var(--rg-border);
                box-shadow: var(--rg-shadow);
                backdrop-filter: blur(18px);
                margin-bottom: {compact_gap};
            }}

            .rg-hero::before {{
                content: "";
                position: absolute;
                inset: -20% auto auto -10%;
                width: 260px;
                height: 260px;
                border-radius: 999px;
                background: radial-gradient(circle, color-mix(in srgb, var(--rg-accent) 45%, transparent) 0%, transparent 70%);
                opacity: {glow_opacity};
                animation: rg-float 9s ease-in-out infinite;
                pointer-events: none;
            }}

            .rg-badge {{
                display: inline-flex;
                align-items: center;
                gap: 0.45rem;
                padding: 0.4rem 0.75rem;
                border-radius: 999px;
                background: color-mix(in srgb, var(--rg-accent) 14%, transparent);
                color: var(--rg-text);
                border: 1px solid color-mix(in srgb, var(--rg-accent) 28%, transparent);
                font-size: 0.82rem;
                font-weight: 700;
                letter-spacing: 0.02em;
            }}

            .rg-title {{
                font-family: 'Sora', sans-serif;
                font-size: clamp(2.2rem, 4vw, 3.6rem);
                font-weight: 800;
                line-height: 1.02;
                margin: 0.8rem 0 0.4rem;
                color: var(--rg-text);
            }}

            .rg-subtitle {{
                max-width: 760px;
                color: var(--rg-muted);
                line-height: 1.7;
                margin-bottom: 0.9rem;
            }}

            .rg-card, [data-testid="stForm"], .stPopover, [data-testid="stMetric"], .stAlert, .stPlotlyChart, .element-container .stMarkdown {{
                border-radius: {radius}px;
            }}

            [data-testid="stForm"], .rg-card {{
                background: linear-gradient(180deg, var(--rg-surface) 0%, color-mix(in srgb, var(--rg-surface-solid) 84%, transparent) 100%);
                border: 1px solid var(--rg-border);
                box-shadow: var(--rg-shadow);
                backdrop-filter: blur(14px);
                padding: 0.35rem 0.35rem 0.6rem 0.35rem;
            }}

            .rg-card-inner {{
                padding: 0.9rem 1rem;
            }}

            .rg-mini-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
                gap: {compact_gap};
                margin-top: 1rem;
            }}

            .rg-mini-card {{
                padding: 1rem;
                border-radius: {radius - 6}px;
                background: linear-gradient(180deg, color-mix(in srgb, var(--rg-accent) 11%, var(--rg-surface-solid)) 0%, var(--rg-surface) 100%);
                border: 1px solid color-mix(in srgb, var(--rg-accent) 18%, var(--rg-border));
                animation: rg-fade-up {motion_seconds} ease-out;
            }}

            .rg-mini-label {{
                color: var(--rg-muted);
                font-size: 0.82rem;
                text-transform: uppercase;
                letter-spacing: 0.08em;
            }}

            .rg-mini-value {{
                margin-top: 0.35rem;
                color: var(--rg-text);
                font-size: 1.28rem;
                font-weight: 800;
            }}

            .rg-floating-credit {{
                position: fixed;
                right: 1rem;
                bottom: 1rem;
                z-index: 9999;
                padding: 0.72rem 1rem;
                border-radius: 999px;
                color: var(--rg-text);
                background: linear-gradient(135deg, color-mix(in srgb, var(--rg-accent) 22%, var(--rg-surface-solid)) 0%, var(--rg-surface) 100%);
                border: 1px solid color-mix(in srgb, var(--rg-accent) 34%, transparent);
                box-shadow: var(--rg-shadow);
                backdrop-filter: blur(12px);
                font-size: 0.9rem;
                font-weight: 700;
            }}

            .rg-section-title {{
                font-family: 'Sora', sans-serif;
                font-size: 1.15rem;
                font-weight: 700;
                margin-bottom: 0.3rem;
            }}

            .rg-section-copy {{
                color: var(--rg-muted);
                margin-bottom: 0.8rem;
            }}

            .stButton > button, .stDownloadButton > button, [data-testid="stFormSubmitButton"] > button {{
                border-radius: 999px !important;
                border: 0 !important;
                padding: 0.82rem 1.25rem !important;
                font-weight: 800 !important;
                color: #06111d !important;
                background: linear-gradient(135deg, var(--rg-accent) 0%, var(--rg-accent-alt) 100%) !important;
                box-shadow: 0 16px 36px color-mix(in srgb, var(--rg-accent) 30%, transparent) !important;
                transition: transform {motion_seconds} ease, box-shadow {motion_seconds} ease !important;
            }}

            .stButton > button:hover, .stDownloadButton > button:hover, [data-testid="stFormSubmitButton"] > button:hover {{
                transform: translateY(-2px);
                box-shadow: 0 22px 44px color-mix(in srgb, var(--rg-accent) 40%, transparent) !important;
            }}

            .stSelectbox label, .stNumberInput label, .stSlider label, .stRadio label, .stFileUploader label, .stCameraInput label, .stCheckbox label {{
                color: var(--rg-text) !important;
                font-weight: 700 !important;
            }}

            .stTextInput input, .stNumberInput input, .stTextArea textarea, .stSelectbox [data-baseweb="select"], .stMultiSelect [data-baseweb="select"] {{
                border-radius: {radius - 10}px !important;
            }}

            .stTextInput input, .stNumberInput input, textarea, [data-baseweb="input"] input, [data-baseweb="select"] > div {{
                background: color-mix(in srgb, var(--rg-surface-solid) 92%, {glass_alpha}) !important;
                color: var(--rg-text) !important;
                border: 1px solid var(--rg-border) !important;
            }}

            .stMarkdown, .stCaption, p, label, span, div {{
                color: inherit;
            }}

            [data-testid="stMetric"] {{
                background: linear-gradient(180deg, var(--rg-surface) 0%, color-mix(in srgb, var(--rg-accent) 6%, var(--rg-surface-solid)) 100%);
                border: 1px solid var(--rg-border);
                box-shadow: var(--rg-shadow);
                padding: 1rem;
            }}

            [data-testid="stMetricLabel"], [data-testid="stMetricValue"] {{
                color: var(--rg-text) !important;
            }}

            .stAlert {{
                background: linear-gradient(180deg, var(--rg-surface) 0%, color-mix(in srgb, var(--rg-accent) 6%, var(--rg-surface-solid)) 100%) !important;
                color: var(--rg-text) !important;
                border: 1px solid var(--rg-border) !important;
            }}

            .rg-divider {{
                height: 1px;
                background: linear-gradient(90deg, transparent, var(--rg-border), transparent);
                margin: 0.85rem 0 1rem;
            }}

            .rg-settings-note {{
                color: var(--rg-muted);
                font-size: 0.88rem;
                line-height: 1.5;
            }}

            @keyframes rg-fade-up {{
                from {{ opacity: 0; transform: translateY(16px); }}
                to {{ opacity: 1; transform: translateY(0); }}
            }}

            @keyframes rg-float {{
                0%, 100% {{ transform: translate3d(0,0,0); }}
                50% {{ transform: translate3d(18px, 18px, 0); }}
            }}
            </style>
            """
        ),
        unsafe_allow_html=True,
    )


def fetch_live_weather():
    try:
        url = "https://api.open-meteo.com/v1/forecast?latitude=23.23&longitude=87.07&current_weather=true"
        weather = requests.get(url, timeout=10).json()["current_weather"]
        return {
            "temperature": weather.get("temperature"),
            "windspeed": weather.get("windspeed"),
            "code": weather.get("weathercode"),
        }
    except Exception:
        return None


def render_metric_cards(items):
    card_html = "".join(
        f'<div class="rg-mini-card">'
        f'<div class="rg-mini-label">{item["label"]}</div>'
        f'<div class="rg-mini-value">{item["value"]}</div>'
        '</div>'
        for item in items
    )
    st.markdown(f'<div class="rg-mini-grid">{card_html}</div>', unsafe_allow_html=True)


model_dl = get_ai_model()
yield_model = get_yield_model()

if "result" not in st.session_state:
    st.session_state.result = None

logo_path = "assets/logo.png"
logo_data_uri = load_logo_data_uri(logo_path)
weather_data = fetch_live_weather()
if "ui_settings" not in st.session_state:
    st.session_state.ui_settings = default_ui_settings()

header_left, header_right = st.columns([7, 2])
with header_right:
    with st.popover("⚙ Settings", use_container_width=True):
        st.session_state.ui_settings["theme_mode"] = st.selectbox(
            "Theme Mode",
            ["System Default", "Dark", "Light"],
            index=["System Default", "Dark", "Light"].index(st.session_state.ui_settings["theme_mode"]),
        )
        st.session_state.ui_settings["accent_palette"] = st.selectbox(
            "Accent Palette",
            ["Aurora", "Ocean", "Sunset", "Royal"],
            index=["Aurora", "Ocean", "Sunset", "Royal"].index(st.session_state.ui_settings["accent_palette"]),
        )
        st.session_state.ui_settings["font_family"] = st.selectbox(
            "Font Family",
            ["Manrope", "Sora", "Poppins"],
            index=["Manrope", "Sora", "Poppins"].index(st.session_state.ui_settings["font_family"]),
        )
        st.session_state.ui_settings["font_scale"] = st.slider("Font Size", 14, 20, st.session_state.ui_settings["font_scale"])
        st.session_state.ui_settings["radius"] = st.slider("Card Roundness", 16, 34, st.session_state.ui_settings["radius"])
        st.session_state.ui_settings["surface_style"] = st.selectbox(
            "Surface Style",
            ["Glass", "Solid"],
            index=["Glass", "Solid"].index(st.session_state.ui_settings["surface_style"]),
        )
        st.session_state.ui_settings["motion"] = st.selectbox(
            "Animation Style",
            ["Dynamic", "Minimal"],
            index=["Dynamic", "Minimal"].index(st.session_state.ui_settings["motion"]),
        )
        st.session_state.ui_settings["layout_density"] = st.selectbox(
            "Layout Density",
            ["Comfortable", "Compact"],
            index=["Comfortable", "Compact"].index(st.session_state.ui_settings["layout_density"]),
        )
        st.session_state.ui_settings["hero_glow"] = st.toggle("Ambient Glow", value=st.session_state.ui_settings["hero_glow"])
        if st.button("Reset Interface", use_container_width=True):
            st.session_state.ui_settings = default_ui_settings()
            st.rerun()
        st.markdown(
            '<div class="rg-settings-note">Your theme setting now controls the whole in-app appearance, so you no longer need to switch Streamlit theme separately.</div>',
            unsafe_allow_html=True,
        )

token_pack = get_theme_tokens(
    st.session_state.ui_settings["theme_mode"],
    st.session_state.ui_settings["accent_palette"],
)
inject_custom_theme(st.session_state.ui_settings, token_pack)

with header_left:
    weather_label = "Weather offline"
    if weather_data and weather_data.get("temperature") is not None:
        weather_label = f"Live weather {weather_data['temperature']}°C"
    st.markdown(
        textwrap.dedent(
            f"""
            <div class="rg-shell">
                <div class="rg-hero">
                    <div class="rg-badge">Precision Rice Intelligence</div>
                    <div class="rg-title">RiceGenixAI</div>
                    <div class="rg-subtitle">
                        Sleek forecasting, disease scanning, growth projection, graphs, and report export in one premium dashboard for farmers and field teams.
                    </div>
                    <div style="display:flex; gap:0.7rem; flex-wrap:wrap; margin-top:0.6rem;">
                        <div class="rg-badge">{weather_label}</div>
                        <div class="rg-badge">Smart growth projection enabled</div>
                        <div class="rg-badge">PDF + AI disease analysis</div>
                    </div>
                </div>
            </div>
            """
        ),
        unsafe_allow_html=True,
    )

st.markdown('<div class="rg-floating-credit">A Project by Aranyak</div>', unsafe_allow_html=True)

if model_dl is None:
    st.info("AI disease model is unavailable, but yield prediction can still run.")

if not SKLEARN_AVAILABLE or yield_model is None:
    st.error("Prediction model failed to load. Please check deployment.")
    st.stop()

main_col, side_col = st.columns([1.4, 0.8], gap="large")

with side_col:
    st.markdown(
        textwrap.dedent(
            """
            <div class="rg-card">
                <div class="rg-card-inner">
                    <div class="rg-section-title">Interface Overview</div>
                    <div class="rg-section-copy">Everything below follows your in-app settings instantly, including light mode, dark mode, accent colors, rounded cards, motion, and typography.</div>
                </div>
            </div>
            """
        ),
        unsafe_allow_html=True,
    )
    overview_items = [
        {"label": "Theme", "value": st.session_state.ui_settings["theme_mode"]},
        {"label": "Accent", "value": st.session_state.ui_settings["accent_palette"]},
        {"label": "Motion", "value": st.session_state.ui_settings["motion"]},
        {"label": "Density", "value": st.session_state.ui_settings["layout_density"]},
    ]
    render_metric_cards(overview_items)
    st.markdown('<div class="rg-divider"></div>', unsafe_allow_html=True)
    if weather_data:
        weather_items = [
            {"label": "Temperature", "value": f"{weather_data['temperature']}°C"},
            {"label": "Wind", "value": f"{weather_data['windspeed']} km/h" if weather_data.get("windspeed") is not None else "N/A"},
            {"label": "Mode", "value": st.session_state.ui_settings["theme_mode"]},
        ]
        render_metric_cards(weather_items)
    st.markdown(
        textwrap.dedent(
            """
            <div class="rg-card" style="margin-top:1rem;">
                <div class="rg-card-inner">
                    <div class="rg-section-title">Support</div>
                    <div class="rg-section-copy">Contact: 9832114844<br>Email: tamal.bot@gmail.com</div>
                </div>
            </div>
            """
        ),
        unsafe_allow_html=True,
    )
    if st.button("Reset Inputs", use_container_width=True):
        preserved = st.session_state.ui_settings
        st.session_state.clear()
        st.session_state.ui_settings = preserved
        st.rerun()

with main_col:
    st.markdown(
        textwrap.dedent(
            """
            <div class="rg-card">
                <div class="rg-card-inner">
                    <div class="rg-section-title">Crop Forecast Workspace</div>
                    <div class="rg-section-copy">Enter the observation month, field conditions, and crop image. The system projects full-growth height and predicts yield before harvest.</div>
                </div>
            </div>
            """
        ),
        unsafe_allow_html=True,
    )

    with st.form("prediction_form"):
        upper_a, upper_b = st.columns(2, gap="large")
        with upper_a:
            crop_name = st.selectbox("Select Rice Variety", list(rice_data.keys()))
            height = st.number_input("Enter plant height (in inches)", min_value=0.0, value=0.0)
            months_observed = st.number_input("Months observed after planting", min_value=0.5, max_value=12.0, value=1.0, step=0.5)
            gene_b = st.radio("Disease resistant?", ["Yes", "No"], horizontal=True)
            gene_c = st.radio("Drought tolerant?", ["Yes", "No"], horizontal=True)

        with upper_b:
            rain = st.number_input("Annual Rainfall (mm)", min_value=0.0, value=0.0)
            temp_unit = st.selectbox("Temperature Unit", ["Celsius", "Fahrenheit"])
            temp_input = st.number_input("Average Temperature", value=0.0)
            soil_type = st.selectbox("Soil Type", ["Loamy", "Clay", "Sandy", "Alluvial", "Laterite"])
            water_source = st.selectbox("Irrigation Water Type", ["Rainwater", "Groundwater", "Mixed"])
            fertilizer_use = st.selectbox("Fertilizer Usage", ["Organic", "Chemical", "Mixed"])

        st.markdown('<div class="rg-divider"></div>', unsafe_allow_html=True)
        soil_a, soil_b = st.columns(2, gap="large")
        with soil_a:
            manual_ph = st.checkbox("I know my soil pH")
            ph_input = st.slider("Enter Soil pH", 3.0, 9.0, 6.5, step=0.1)
        with soil_b:
            uploaded_file = st.file_uploader("Upload rice crop image", type=["jpg", "jpeg", "png"])
            camera_file = st.camera_input("Or use your phone camera")

        submitted = st.form_submit_button("Predict Yield", use_container_width=True)

preview_image, selected_image_source = resolve_image_input(uploaded_file, camera_file)
if preview_image is not None:
    st.image(preview_image, caption="Uploaded Image", use_container_width=True)

current_signature = build_input_signature(
    crop_name, height, months_observed, gene_b, gene_c, rain, temp_unit, temp_input,
    soil_type, water_source, fertilizer_use, manual_ph, ph_input, preview_image is not None
)

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
    expected_height = rice_data[crop_name]["height"]
    growth_metrics = project_growth_metrics(crop_name, height_val, months_observed)
    projected_final_height = growth_metrics["projected_final_height"]
    expected_height_now = growth_metrics["expected_height_now"]
    maturity_months = growth_metrics["maturity_months"]
    g4 = 1 if height_val >= expected_height_now * 1.05 or projected_final_height >= expected_height * 1.05 else 0
    g1 = 1 if projected_final_height >= expected_height else 0

    disease_name = "Not Checked"
    image_for_report = None
    if preview_image is not None:
        try:
            image_for_prediction = preview_image.copy()
            disease_name = predict(model_dl, image_for_prediction)
            image_for_report = image_for_prediction.copy()
        except Exception:
            disease_name = "Model Error"

    base_pred = yield_model.predict([[g1, g2, g3, g4, rain_val, temp_val, ph]])[0]
    disease, water = crop_health(g2, rain_val, temp_val)
    final_pred = float(base_pred)

    if projected_final_height < expected_height:
        final_pred -= 0.7
        height_flag = "Projected full growth is below standard -> yield may decrease."
    elif projected_final_height > expected_height + 10:
        final_pred -= 0.3
        height_flag = "Projected overgrowth detected -> possible nutrient imbalance."
    else:
        height_flag = "Projected full growth looks normal."

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
        "months_observed": float(months_observed),
        "maturity_months": maturity_months,
        "expected_height_now": expected_height_now,
        "projected_final_height": projected_final_height,
        "expected_height": expected_height,
        "uploaded_image_path": uploaded_image_path,
        "signature": current_signature,
        "theme_tokens": token_pack["active"],
    }

if st.session_state.result and st.session_state.result.get("signature") == current_signature:
    res = st.session_state.result
    disease_val = res.get("disease", 0)
    water_val = res.get("water", 0)
    st.markdown(
        """
        <div class="rg-card" style="margin-top:1rem;">
            <div class="rg-card-inner">
                <div class="rg-section-title">Prediction Summary</div>
                <div class="rg-section-copy">High-clarity forecast cards, projected final growth, and field-ready AI observations.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    render_metric_cards(
        [
            {"label": "Predicted Yield", "value": f"{res['yield']:.2f} t/ha"},
            {"label": "Detected Disease", "value": res["disease_name"]},
            {"label": "Projected Final Height", "value": f"{res['projected_final_height']:.2f} in"},
            {"label": "Soil pH", "value": f"{res['ph']:.2f}"},
            {"label": "Observation Month", "value": f"{res['months_observed']:.1f}"},
            {"label": "Full Growth Time", "value": f"{res['maturity_months']:.1f} mo"},
        ]
    )

    summary_a, summary_b = st.columns([1, 1], gap="large")
    with summary_a:
        st.markdown("### Growth Projection")
        st.write(f"Crop Selected: {res['crop_name']}")
        st.write(f"Expected Standard Height: {res['expected_height']} inches")
        st.write(f"Expected Height At This Month: {res['expected_height_now']:.2f} inches")
        st.write(f"Your Crop Height: {res['input_height']} inches")
        st.write(f"Predicted Final Height: {res['projected_final_height']:.2f} inches")
        st.write(res["height_flag"])
        st.markdown("### Gene Representation")
        st.json(
            {
                "Gene_A": res["genes"][0],
                "Gene_B": res["genes"][1],
                "Gene_C": res["genes"][2],
                "Gene_D": res["genes"][3],
            }
        )

    with summary_b:
        st.markdown("### Crop Health Analysis")
        st.write(f"Disease Risk: {'High' if disease_val else 'Low'}")
        st.write(f"Water Stress: {'Yes' if water_val else 'No'}")
        st.write(f"Detected Disease: {res['disease_name']}")
        st.markdown("### AI Suggestions")
        if res["disease_name"] not in {"Healthy", "Not Checked"}:
            st.write(f"Treatment: {res['treatment']}")
            st.write("Control disease early to prevent major yield loss.")
        elif res["disease_name"] == "Healthy":
            st.write(res["treatment"])

        if res["projected_final_height"] < res["expected_height"]:
            st.write("Projected growth is below expected -> improve nitrogen supply and irrigation.")
        elif res["projected_final_height"] > res["expected_height"] + 10:
            st.write("Projected excessive growth -> reduce nitrogen and balance nutrients.")

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

    st.markdown("### Data Analysis")
    fig1, fig2 = generate_graphs(res)
    graph_a, graph_b = st.columns(2, gap="large")
    with graph_a:
        st.pyplot(fig1, use_container_width=True)
    with graph_b:
        st.pyplot(fig2, use_container_width=True)
    pdf_bytes = build_pdf_report(res, fig1, fig2, logo_path)
    pdf_b64 = base64.b64encode(pdf_bytes).decode()
    st.download_button(
        label="Download PDF Report",
        data=pdf_bytes,
        file_name="RiceGenix_Report.pdf",
        mime="application/pdf",
        key="pdf_download",
        use_container_width=True,
    )
    st.markdown(
        f'<a href="data:application/pdf;base64,{pdf_b64}" download="RiceGenix_Report.pdf">If mobile download fails, tap here</a>',
        unsafe_allow_html=True,
    )
