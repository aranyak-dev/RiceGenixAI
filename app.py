import os
import tempfile
import base64
import textwrap
import copy
import io
import math
import struct
import time
import wave

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
    "Gobindobhog": {
        "height": 51,
        "height_range": (48, 51, 55),
        "drought": 0,
        "disease": 1,
        "maturity_months": 4.0,
        "maturity_range": (3.8, 4.5),
    },
    "Swarna": {
        "height": 39,
        "height_range": (36, 39, 43),
        "drought": 0,
        "disease": 1,
        "maturity_months": 4.0,
        "maturity_range": (3.8, 4.4),
    },
    "Swarna Sub1": {
        "height": 39,
        "height_range": (36, 39, 42),
        "drought": 0,
        "disease": 1,
        "maturity_months": 4.0,
        "maturity_range": (3.8, 4.4),
    },
    "IR36": {
        "height": 35,
        "height_range": (32, 35, 39),
        "drought": 0,
        "disease": 1,
        "maturity_months": 3.8,
        "maturity_range": (3.5, 4.2),
    },
    "IR64": {
        "height": 39,
        "height_range": (36, 39, 43),
        "drought": 1,
        "disease": 1,
        "maturity_months": 4.0,
        "maturity_range": (3.8, 4.4),
    },
    "Minikit": {
        "height": 35,
        "height_range": (32, 35, 39),
        "drought": 0,
        "disease": 1,
        "maturity_months": 3.8,
        "maturity_range": (3.6, 4.2),
    },
    "Banskathi": {
        "height": 55,
        "height_range": (50, 55, 60),
        "drought": 1,
        "disease": 0,
        "maturity_months": 4.8,
        "maturity_range": (4.4, 5.3),
    },
    "Tulaipanji": {
        "height": 55,
        "height_range": (50, 55, 60),
        "drought": 0,
        "disease": 1,
        "maturity_months": 5.0,
        "maturity_range": (4.6, 5.4),
    },
    "Kalijeera": {
        "height": 50,
        "height_range": (47, 50, 54),
        "drought": 0,
        "disease": 1,
        "maturity_months": 4.8,
        "maturity_range": (4.5, 5.1),
    },
    "Radhatilak": {
        "height": 52,
        "height_range": (48, 52, 56),
        "drought": 0,
        "disease": 1,
        "maturity_months": 4.8,
        "maturity_range": (4.5, 5.1),
    },
    "Dudheswar": {
        "height": 50,
        "height_range": (46, 50, 54),
        "drought": 0,
        "disease": 1,
        "maturity_months": 4.8,
        "maturity_range": (4.5, 5.1),
    },
    "Kataribhog": {
        "height": 48,
        "height_range": (44, 48, 52),
        "drought": 0,
        "disease": 1,
        "maturity_months": 4.8,
        "maturity_range": (4.5, 5.1),
    },
    "Radhunipagal": {
        "height": 52,
        "height_range": (48, 52, 56),
        "drought": 0,
        "disease": 1,
        "maturity_months": 4.8,
        "maturity_range": (4.5, 5.1),
    },
    "Badshabhog": {
        "height": 50,
        "height_range": (46, 50, 54),
        "drought": 0,
        "disease": 1,
        "maturity_months": 4.8,
        "maturity_range": (4.5, 5.1),
    },
}

LANGUAGE_OPTIONS = ["English", "Bengali", "Hindi"]
translations = {
    "English": {
        "settings_title": "⚙ Settings",
        "theme_mode": "Theme Mode",
        "accent_palette": "Accent Palette",
        "theme_system_default": "System Default",
        "theme_dark": "Dark",
        "theme_light": "Light",
        "surface_glass": "Glass",
        "surface_solid": "Solid",
        "motion_dynamic": "Dynamic",
        "motion_minimal": "Minimal",
        "density_comfortable": "Comfortable",
        "density_compact": "Compact",
        "font_family": "Font Family",
        "font_size": "Font Size",
        "card_roundness": "Card Roundness",
        "surface_style": "Surface Style",
        "animation_style": "Animation Style",
        "layout_density": "Layout Density",
        "ambient_glow": "Ambient Glow",
        "language": "Language",
        "reset_interface": "Reset Interface",
        "settings_note": "Your theme setting now controls the whole in-app appearance, so you no longer need to switch Streamlit theme separately.",
        "precision_banner": "Precision Rice Intelligence",
        "hero_subtitle": "Sleek forecasting, disease scanning, growth projection, graphs, and report export in one premium dashboard for farmers and field teams.",
        "home_welcome": "Welcome",
        "home_subtitle": "Launch RiceGenixAI in your preferred language and begin smarter rice decisions.",
        "home_choose_language": "Choose your language",
        "start_button": "Start in English",
        "badge_weather_offline": "Weather offline",
        "live_weather": "Live weather {temperature}°C",
        "badge_smart_projection": "Smart growth projection enabled",
        "badge_pdf_ai": "PDF + AI disease analysis",
        "floating_credit": "A Project by Aranyak",
        "ai_unavailable_info": "AI disease model is unavailable, but yield prediction can still run.",
        "prediction_model_error": "Prediction model failed to load. Please check deployment.",
        "interface_overview_title": "Interface Overview",
        "interface_overview_copy": "Everything below follows your in-app settings instantly, including light mode, dark mode, accent colors, rounded cards, motion, and typography.",
        "support_title": "Support",
        "support_copy": "Contact: 9832114844<br>Email: tamal.bot@gmail.com",
        "reset_inputs": "Reset Inputs",
        "theme_label": "Theme",
        "accent_label": "Accent",
        "motion_label": "Motion",
        "density_label": "Density",
        "temperature_label": "Temperature",
        "wind_label": "Wind",
        "mode_label": "Mode",
        "workspace_title": "Crop Forecast Workspace",
        "workspace_copy": "Enter the observation month, field conditions, and crop image. The system projects full-growth height and predicts yield before harvest.",
        "select_rice_variety": "Select Rice Variety",
        "enter_plant_height": "Enter plant height (in inches)",
        "months_observed": "Months observed after planting",
        "disease_resistant": "Disease resistant?",
        "drought_tolerant": "Drought tolerant?",
        "annual_rainfall": "Annual Rainfall (mm)",
        "temperature_unit": "Temperature Unit",
        "average_temperature": "Average Temperature",
        "soil_type": "Soil Type",
        "irrigation_water_type": "Irrigation Water Type",
        "fertilizer_usage": "Fertilizer Usage",
        "i_know_my_soil_ph": "I know my soil pH",
        "enter_soil_ph": "Enter Soil pH",
        "upload_crop_image": "Upload rice crop image",
        "use_phone_camera": "Or use your phone camera",
        "predict_yield": "Predict Yield",
        "invalid_numeric_values": "Please enter valid numeric values for rainfall, temperature, and height.",
        "prediction_summary_title": "Prediction Summary",
        "prediction_summary_copy": "High-clarity forecast cards, projected final growth, and field-ready AI observations.",
        "growth_projection": "Growth Projection",
        "crop_selected": "Crop Selected: {crop_name}",
        "expected_standard_height": "Expected Standard Height: {height} inches",
        "expected_height_at_month": "Expected Height At This Month: {height} inches",
        "your_crop_height": "Your Crop Height: {height} inches",
        "predicted_final_height": "Predicted Final Height: {height} inches",
        "gene_representation": "Gene Representation",
        "crop_health_analysis": "Crop Health Analysis",
        "disease_risk": "Disease Risk: {value}",
        "water_stress": "Water Stress: {value}",
        "detected_disease": "Detected Disease: {value}",
        "ai_suggestions": "AI Suggestions",
        "treatment_recommendation": "Treatment: {value}",
        "control_disease_early": "Control disease early to prevent major yield loss.",
        "projected_growth_below": "Projected growth is below expected -> improve nitrogen supply and irrigation.",
        "projected_excess_growth": "Projected excessive growth -> reduce nitrogen and balance nutrients.",
        "soil_acidic": "Soil is acidic -> add lime or dolomite to increase pH.",
        "use_organic_compost": "Use organic compost to stabilize soil condition.",
        "soil_alkaline": "Soil is alkaline -> add gypsum or organic matter.",
        "avoid_excessive_chemicals": "Avoid excessive chemical fertilizers.",
        "very_low_rainfall": "Very low rainfall -> increase irrigation frequency.",
        "moderately_low_rainfall": "Moderately low rainfall -> maintain regular irrigation.",
        "excess_rainfall": "Excess rainfall -> improve drainage system.",
        "high_temperature_stress": "High temperature stress -> maintain water level in the field.",
        "low_temperature_slow": "Low temperature may slow growth, so monitor the crop carefully.",
        "use_high_yield_seeds": "Use high-yield variety seeds like Swarna or IR64.",
        "apply_balanced_npk": "Apply balanced NPK fertilizers based on soil testing.",
        "monitor_weekly": "Monitor the crop weekly for early disease detection.",
        "use_proper_spacing": "Use proper spacing to avoid fungal infection.",
        "data_analysis": "Data Analysis",
        "rainfall_impact_title": "Rainfall Impact on Yield",
        "temperature_impact_title": "Temperature Impact on Yield",
        "download_pdf_report": "Download PDF Report",
        "mobile_download_fails": "If mobile download fails, tap here",
        "yes": "Yes",
        "no": "No",
        "celsius": "Celsius",
        "fahrenheit": "Fahrenheit",
        "loamy": "Loamy",
        "clay": "Clay",
        "sandy": "Sandy",
        "alluvial": "Alluvial",
        "laterite": "Laterite",
        "rainwater": "Rainwater",
        "groundwater": "Groundwater",
        "mixed": "Mixed",
        "organic": "Organic",
        "chemical": "Chemical",
        "language_english": "English",
        "language_bengali": "বাংলা",
        "language_hindi": "हिन्दी",
    },
    "Bengali": {
        "settings_title": "⚙ সেটিংস",
        "theme_mode": "থিম মোড",
        "accent_palette": "হাইলাইট রঙ",
        "theme_system_default": "সিস্টেম ডিফল্ট",
        "theme_dark": "ডার্ক",
        "theme_light": "লাইট",
        "surface_glass": "গ্লাস",
        "surface_solid": "সলিড",
        "motion_dynamic": "ডায়নামিক",
        "motion_minimal": "মিনিমাল",
        "density_comfortable": "কমফর্টেবল",
        "density_compact": "কম্প্যাক্ট",
        "font_family": "ফন্ট পরিবার",
        "font_size": "ফন্ট সাইজ",
        "card_roundness": "কার্ড গোলাকৃতি",
        "surface_style": "প্যানেল স্টাইল",
        "animation_style": "অ্যানিমেশন স্টাইল",
        "layout_density": "লেআউট ঘনত্ব",
        "ambient_glow": "হালকা আলো",
        "language": "ভাষা",
        "reset_interface": "ইন্টারফেস রিসেট করুন",
        "settings_note": "আপনার থিম সেটিং এখন সম্পূর্ণ অ্যাপের চেহারা নিয়ন্ত্রণ করে।",
        "precision_banner": "প্রিসিশন রাইস ইন্টেলিজেন্স",
        "hero_subtitle": "ফার্মার এবং ফিল্ড টিমের জন্য পূর্বাভাস, রোগ স্ক্যান, বৃদ্ধি পূর্বাভাস, গ্রাফ এবং রিপোর্ট একক ড্যাশবোর্ডে।",
        "home_welcome": "স্বাগতম",
        "home_subtitle": "আপনার ভাষা নির্বাচন করুন এবং RiceGenixAI দিয়ে দ্রুত শুরু করুন।",
        "home_choose_language": "আপনার ভাষা নির্বাচন করুন",
        "start_button": "বাংলা ভাষায় শুরু করুন",
        "badge_weather_offline": "ওয়েদার অফলাইন",
        "live_weather": "লাইভ ওয়েদার {temperature}°C",
        "badge_smart_projection": "স্মার্ট বৃদ্ধি পূর্বাভাস সক্রিয়",
        "badge_pdf_ai": "পিডিএফ + এআই রোগ বিশ্লেষণ",
        "floating_credit": "আরন্যকের একটি প্রোজেক্ট",
        "ai_unavailable_info": "এআই রোগ মডেল পাওয়া যায়নি, তবে উৎপাদন পূর্বাভাস এখনও চলতে পারে।",
        "prediction_model_error": "পূর্বাভাস মডেল লোড হয়নি। দয়া করে ডিপ্লয়মেন্ট চেক করুন।",
        "interface_overview_title": "ইন্টারফেস ওভারভিউ",
        "interface_overview_copy": "নিচের সমস্ত কিছু আপনার ইন-অ্যাপ সেটিংস অনুসারে কাজ করে, লাইট/ডার্ক মোড, রঙ, কার্ড, মোশন এবং টাইপোগ্রাফি সহ।",
        "support_title": "সাপোর্ট",
        "support_copy": "যোগাযোগ: 9832114844<br>ইমেইল: tamal.bot@gmail.com",
        "reset_inputs": "ইনপুটস রিসেট করুন",
        "theme_label": "থিম",
        "accent_label": "রঙ",
        "motion_label": "মোশন",
        "density_label": "ঘনত্ব",
        "temperature_label": "তাপমাত্রা",
        "wind_label": "হাওয়া",
        "mode_label": "মোড",
        "workspace_title": "কৃষি ভবিষ্যদ্বাণী কর্মক্ষেত্র",
        "workspace_copy": "পর্যবেক্ষণ মাস, ক্ষেত্রের পরিস্থিতি এবং ফসলের ছবি দিন। সিস্টেম পূর্ণ বৃদ্ধি উচ্চতা ও শস্য ফলন পূর্বাভাস করে।",
        "select_rice_variety": "চালের জাত নির্বাচন করুন",
        "enter_plant_height": "উদ্ভিদের উচ্চতা লিখুন (ইঞ্চিতে)",
        "months_observed": "রোপণের পর কত মাস পর্যবেক্ষণ করা হয়েছে",
        "disease_resistant": "রোগ প্রতিরোধী?",
        "drought_tolerant": "শুকিয়ে সহনশীল?",
        "annual_rainfall": "বার্ষিক বৃষ্টি (মিমি)",
        "temperature_unit": "তাপমাত্রার একক",
        "average_temperature": "গড় তাপমাত্রা",
        "soil_type": "মাটির ধরন",
        "irrigation_water_type": "সেচের জল ধরনের",
        "fertilizer_usage": "সার ব্যবহারের ধরন",
        "i_know_my_soil_ph": "আমি আমার মাটি পিএইচ জানি",
        "enter_soil_ph": "মাটি পিএইচ লিখুন",
        "upload_crop_image": "চাল ফসলের ছবি আপলোড করুন",
        "use_phone_camera": "অথবা আপনার ফোন ক্যামেরা ব্যবহার করুন",
        "predict_yield": "ফলন পূর্বাভাস করুন",
        "invalid_numeric_values": "বৃষ্টিপাত, তাপমাত্রা এবং উচ্চতার জন্য সঠিক সংখ্যা প্রদান করুন।",
        "prediction_summary_title": "পূর্বাভাস সংক্ষিপ্তসার",
        "prediction_summary_copy": "উচ্চ স্পষ্টতার পূর্বাভাস কার্ড, পূর্ণ বৃদ্ধি এবং মাঠ-প্রস্তুত এআই পর্যবেক্ষণ।",
        "growth_projection": "উদ্ভাবন পূর্বাভাস",
        "crop_selected": "নির্বাচিত ফসল: {crop_name}",
        "expected_standard_height": "প্রত্যাশিত মানদণ্ড উচ্চতা: {height} ইঞ্চি",
        "expected_height_at_month": "এই মাসে প্রত্যাশিত উচ্চতা: {height} ইঞ্চি",
        "your_crop_height": "আপনার ফসলের উচ্চতা: {height} ইঞ্চি",
        "predicted_final_height": "প্রাক্কলিত চূড়ান্ত উচ্চতা: {height} ইঞ্চি",
        "gene_representation": "জিন রিপ্রেজেন্টেশন",
        "crop_health_analysis": "ফসলের স্বাস্থ্যের বিশ্লেষণ",
        "disease_risk": "রোগ ঝুঁকি: {value}",
        "water_stress": "জল চাপ: {value}",
        "detected_disease": "চিহ্নিত রোগ: {value}",
        "ai_suggestions": "এআই পরামর্শ",
        "treatment_recommendation": "চিকিৎসা: {value}",
        "control_disease_early": "প্রধান ফলন ক্ষতি রোধে দ্রুত রোগ নিয়ন্ত্রণ করুন।",
        "projected_growth_below": "প্রকল্পিত বৃদ্ধি প্রত্যাশার নীচে -> নাইট্রোজেন এবং সেচ উন্নত করুন।",
        "projected_excess_growth": "প্রকল্পিত অতিরিক্ত বৃদ্ধি -> সার এবং পুষ্টি সামঞ্জস্য করুন।",
        "soil_acidic": "মাটি অ্যাসিডিক -> পিএইচ বাড়াতে চুন বা ডোলোমাইট যোগ করুন।",
        "use_organic_compost": "মাটি স্থিতিশীল করার জন্য জৈব কম্পোস্ট ব্যবহার করুন।",
        "soil_alkaline": "মাটি ক্ষারীয় -> জিপসাম বা জৈব পদার্থ যোগ করুন।",
        "avoid_excessive_chemicals": "অতিরিক্ত রাসায়নিক সার এড়িয়ে চলুন।",
        "very_low_rainfall": "খুব কম বৃষ্টি -> সেচের ফ্রিকোয়েন্সি বাড়ান।",
        "moderately_low_rainfall": "মাঝারি কম বৃষ্টি -> নিয়মিত সেচ বজায় রাখুন।",
        "excess_rainfall": "অতিরিক্ত বৃষ্টি -> নিষ্কাশন ব্যবস্থা উন্নত করুন।",
        "high_temperature_stress": "উচ্চ তাপমাত্রা চাপ -> ক্ষেত্রে জলস্তর বজায় রাখুন।",
        "low_temperature_slow": "নিম্ন তাপমাত্রা বৃদ্ধির গতি ধীর করতে পারে, সুতরাং মনিটর করুন।",
        "use_high_yield_seeds": "উচ্চ ফলনশীল জাতের বীজ ব্যবহার করুন যেমন Swarna বা IR64।",
        "apply_balanced_npk": "মাটি পরীক্ষার ভিত্তিতে ভারসাম্যপূর্ণ এনপিকে সার ব্যবহার করুন।",
        "monitor_weekly": "সাপ্তাহিকভাবে রোগ নির্ধারণের জন্য ফসল পর্যবেক্ষণ করুন।",
        "use_proper_spacing": "ছত্রাক সংক্রমণ এড়াতে সঠিক দূরত্ব বজায় রাখুন।",
        "data_analysis": "ডেটা বিশ্লেষণ",
        "rainfall_impact_title": "বৃষ্টিপাতের প্রভাব ফলনে",
        "temperature_impact_title": "তাপমাত্রার প্রভাব ফলনে",
        "download_pdf_report": "পিডিএফ রিপোর্ট ডাউনলোড করুন",
        "mobile_download_fails": "মোবাইল ডাউনলোড ব্যর্থ হলে এখানে চাপুন",
        "yes": "হ্যাঁ",
        "no": "না",
        "celsius": "সেলসিয়াস",
        "fahrenheit": "ফারেনহাইট",
        "loamy": "লোয়ামি",
        "clay": "কাদামাটি",
        "sandy": "বালি মাটি",
        "alluvial": "সমভূমি",
        "laterite": "ল্যাটেরাইট",
        "rainwater": "বৃষ্টির জল",
        "groundwater": "ভূগর্ভ জল",
        "mixed": "মিশ্রিত",
        "organic": "জৈব",
        "chemical": "রাসায়নিক",
        "language_english": "ইংরেজি",
        "language_bengali": "বাংলা",
        "language_hindi": "হিন্দি",
    },
    "Hindi": {
        "settings_title": "⚙ सेटिंग्स",
        "theme_mode": "थीम मोड",
        "accent_palette": "रंग पैलेट",
        "theme_system_default": "System Default",
        "theme_dark": "Dark",
        "theme_light": "Light",
        "surface_glass": "Glass",
        "surface_solid": "Solid",
        "motion_dynamic": "Dynamic",
        "motion_minimal": "Minimal",
        "density_comfortable": "Comfortable",
        "density_compact": "Compact",
        "font_family": "फ़ॉन्ट परिवार",
        "font_size": "फ़ॉन्ट आकार",
        "card_roundness": "कार्ड गोलाई",
        "surface_style": "सतह शैली",
        "animation_style": "एनीमेशन शैली",
        "layout_density": "लेआउट घनत्व",
        "ambient_glow": "आस पास चमक",
        "language": "भाषा",
        "reset_interface": "इंटरफ़ेस रीसेट करें",
        "settings_note": "आपकी थीम सेटिंग अब पूरे ऐप की उपस्थिति नियंत्रित करती है।",
        "precision_banner": "प्रिसिजन राइस इंटेलिजेंस",
        "hero_subtitle": "किसानों और फील्ड टीमों के लिए पूर्वानुमान, रोग स्कैन, वृद्धि प्रक्षेपण, ग्राफ और रिपोर्ट एक्सपोर्ट एक प्रीमियम डैशबोर्ड में।",
        "home_welcome": "स्वागत है",
        "home_subtitle": "अपनी भाषा चुनें और RiceGenixAI के साथ तुरंत शुरू करें।",
        "home_choose_language": "अपनी भाषा चुनें",
        "start_button": "हिंदी में शुरू करें",
        "badge_weather_offline": "मौसम ऑफ़लाइन",
        "live_weather": "लाइव मौसम {temperature}°C",
        "badge_smart_projection": "स्मार्ट वृद्धि प्रक्षेपण सक्षम",
        "badge_pdf_ai": "पीडीएफ + एआई रोग विश्लेषण",
        "floating_credit": "अरन्याक का प्रोजेक्ट",
        "ai_unavailable_info": "एआई रोग मॉडल अनुपलब्ध है, लेकिन फ़सल अनुमान अभी भी चल सकता है।",
        "prediction_model_error": "पूर्वानुमान मॉडल लोड नहीं हुआ। कृपया डिप्लॉयमेंट जाँचें।",
        "interface_overview_title": "इंटरफ़ेस अवलोकन",
        "interface_overview_copy": "नीचे की सभी चीज़ें आपके इन-ऐप सेटिंग्स के अनुसार तुरंत काम करती हैं, जिसमें लाइट/डार्क मोड, रंग, गोल कार्ड, मोशन और टाइपोग्राफी शामिल हैं।",
        "support_title": "सहायता",
        "support_copy": "संपर्क: 9832114844<br>ईमेल: tamal.bot@gmail.com",
        "reset_inputs": "इनपुट रीसेट करें",
        "theme_label": "थीम",
        "accent_label": "रंग",
        "motion_label": "मोशन",
        "density_label": "घनत्व",
        "temperature_label": "तापमान",
        "wind_label": "हवा",
        "mode_label": "मोड",
        "workspace_title": "फसल पूर्वानुमान कार्यक्षेत्र",
        "workspace_copy": "पर्यवेक्षण महीना, खेत की स्थिति और फसल की तस्वीर दर्ज करें। सिस्टम पूर्ण वृद्धि ऊंचाई और उत्पादन की भविष्यवाणी करता है।",
        "select_rice_variety": "चावल की किस्म चुनें",
        "enter_plant_height": "पौधे की ऊंचाई दर्ज करें (इंच में)",
        "months_observed": "रोपण के बाद कितने महीने देखा गया",
        "disease_resistant": "रोग प्रतिरोधी?",
        "drought_tolerant": "सूखा सहिष्णु?",
        "annual_rainfall": "वार्षिक वर्षा (मिमी)",
        "temperature_unit": "तापमान इकाई",
        "average_temperature": "औसत तापमान",
        "soil_type": "मिट्टी का प्रकार",
        "irrigation_water_type": "सिंचाई पानी का प्रकार",
        "fertilizer_usage": "उर्वरक उपयोग",
        "i_know_my_soil_ph": "मुझे अपनी मिट्टी का पीएच पता है",
        "enter_soil_ph": "मिट्टी का पीएच दर्ज करें",
        "upload_crop_image": "चावल की फसल की छवि अपलोड करें",
        "use_phone_camera": "या अपने फ़ोन कैमरा का उपयोग करें",
        "predict_yield": "उपज पूर्वानुमान करें",
        "invalid_numeric_values": "वर्षा, तापमान और ऊँचाई के लिए मान्य संख्यात्मक मान दर्ज करें।",
        "prediction_summary_title": "पूर्वानुमान सारांश",
        "prediction_summary_copy": "उच्च स्पष्टता पूर्वानुमान कार्ड, पूर्ण वृद्धि और फ़ील्ड-तैयार एआई अवलोकन।",
        "growth_projection": "वृद्धि प्रक्षेपण",
        "crop_selected": "चयनित फसल: {crop_name}",
        "expected_standard_height": "अपेक्षित मानक ऊँचाई: {height} इंच",
        "expected_height_at_month": "इस महीने अपेक्षित ऊँचाई: {height} इंच",
        "your_crop_height": "आपकी फसल की ऊँचाई: {height} इंच",
        "predicted_final_height": "पूर्वानुमानित अंतिम ऊँचाई: {height} इंच",
        "gene_representation": "जीन प्रतिनिधित्व",
        "crop_health_analysis": "फसल स्वास्थ्य विश्लेषण",
        "disease_risk": "रोग जोखिम: {value}",
        "water_stress": "जल तनाव: {value}",
        "detected_disease": "पहचाना गया रोग: {value}",
        "ai_suggestions": "एआई सुझाव",
        "treatment_recommendation": "उपचार: {value}",
        "control_disease_early": "मुख्य उपज हानि रोकने के लिए जल्दी रोग नियंत्रित करें।",
        "projected_growth_below": "पूर्वानुमानित वृद्धि अपेक्षा से नीचे -> नाइट्रोजन और सिंचाई बेहतर करें।",
        "projected_excess_growth": "पूर्वानुमानित अत्यधिक वृद्धि -> पोषक तत्व संतुलित करें।",
        "soil_acidic": "मिट्टी अम्लीय है -> पीएच बढ़ाने के लिए चूना या डोलोमाइट डालें।",
        "use_organic_compost": "मिट्टी स्थिर करने के लिए जैविक कंपोस्ट का उपयोग करें।",
        "soil_alkaline": "मिट्टी क्षारीय है -> जिप्सम या जैविक पदार्थ जोड़ें।",
        "avoid_excessive_chemicals": "अत्यधिक रासायनिक उर्वरक से बचें।",
        "very_low_rainfall": "बहुत कम वर्षा -> सिंचाई की आवृत्ति बढ़ाएं।",
        "moderately_low_rainfall": "मध्यम रूप से कम वर्षा -> नियमित सिंचाई बनाए रखें।",
        "excess_rainfall": "अत्यधिक वर्षा -> जल निकासी सिस्टम बेहतर बनाएं।",
        "high_temperature_stress": "उच्च तापमान तनाव -> खेत में जल स्तर बनाए रखें।",
        "low_temperature_slow": "निम्न तापमान वृद्धि धीमा कर सकता है, सावधानी से मॉनिटर करें।",
        "use_high_yield_seeds": "उच्च उपज वाली किस्में जैसे Swarna या IR64 उपयोग करें।",
        "apply_balanced_npk": "मिट्टी परीक्षण के आधार पर संतुलित एनपीके उर्वरक लागू करें।",
        "monitor_weekly": "प्रारम्भिक रोग पहचान के लिये साप्ताहिक रूप से फसल का निरीक्षण करें।",
        "use_proper_spacing": "फंगल संक्रमण से बचने के लिये उचित दूरी का उपयोग करें।",
        "data_analysis": "डेटा विश्लेषण",
        "rainfall_impact_title": "वर्षा का उत्पादन पर प्रभाव",
        "temperature_impact_title": "तापमान का उत्पादन पर प्रभाव",
        "download_pdf_report": "पीडीएफ रिपोर्ट डाउनलोड करें",
        "mobile_download_fails": "यदि मोबाइल डाउनलोड विफल हो, तो यहां दबाएं",
        "yes": "हाँ",
        "no": "नहीं",
        "celsius": "सेल्सियस",
        "fahrenheit": "फारेनहाइट",
        "loamy": "लोअमी",
        "clay": "मिट्टी",
        "sandy": "रेतीली",
        "alluvial": "अलुवियल",
        "laterite": "लेटेराइट",
        "rainwater": "वर्षाजल",
        "groundwater": "भूजल",
        "mixed": "मिश्रित",
        "organic": "कार्बनिक",
        "chemical": "रासायनिक",
        "language_english": "English",
        "language_bengali": "Bengali",
        "language_hindi": "Hindi",
    },
}

def t(key: str, **kwargs):
    language = st.session_state.get("ui_settings", {}).get("language", "English") if "ui_settings" in st.session_state else "English"
    text = translations.get(language, translations["English"]).get(key, translations["English"].get(key, key))
    return text.format(**kwargs)

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


def classify_height_status(crop_info, actual_height, expected_height_now):
    min_height, target_height, max_height = crop_info["height_range"]
    if actual_height >= max_height or actual_height >= expected_height_now * 1.08:
        return "Fast growth — crop is ahead of the expected curve."
    if actual_height >= target_height:
        return "On track — crop is within the strong growth range."
    if actual_height >= max(min_height, expected_height_now * 0.95):
        return "Healthy growth — still within the good growth window."
    if actual_height >= max(min_height * 0.95, expected_height_now * 0.80):
        return "Slightly delayed — the crop can still recover with good nutrition."
    return "Slow growth — needs closer monitoring and support."


def project_growth_metrics(crop_name, actual_height, months_observed):
    crop_info = rice_data[crop_name]
    target_height = float(crop_info["height"])
    min_maturity, max_maturity = crop_info["maturity_range"]
    avg_maturity = float(crop_info["maturity_months"])
    observed_months = max(0.5, float(months_observed))

    # Use a flexible maturity window so growth projection can adapt to early or late maturing crops.
    maturity_fraction = min(max(observed_months / avg_maturity, 0.12), 1.2)
    expected_height_now = target_height * min(maturity_fraction, 1.0)
    expected_height_now = max(expected_height_now, target_height * 0.08)

    raw_rate = actual_height / max(expected_height_now, 0.1)
    if raw_rate >= 1.15:
        projected_final_height = min(target_height * 1.05, actual_height + (target_height - actual_height) * 0.2)
    elif raw_rate >= 1.0:
        projected_final_height = min(target_height * 1.03, actual_height + (target_height - actual_height) * 0.35)
    elif raw_rate >= 0.9:
        projected_final_height = min(target_height * 1.02, actual_height + (target_height - actual_height) * 0.55)
    else:
        projected_final_height = min(max(target_height, actual_height + (target_height - actual_height) * 0.8), target_height * 1.05)

    projected_final_height = max(actual_height, projected_final_height)
    height_status = classify_height_status(crop_info, actual_height, expected_height_now)
    growth_ratio = min(expected_height_now / target_height, 1.0)

    return {
        "maturity_months": avg_maturity,
        "min_maturity_months": min_maturity,
        "max_maturity_months": max_maturity,
        "expected_height_now": expected_height_now,
        "projected_final_height": projected_final_height,
        "growth_ratio": growth_ratio,
        "height_status": height_status,
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
    content.append(Spacer(1, 8))
    content.append(Paragraph("<i>RiceGenixAI can make mistakes.</i>", styles["Italic"]))
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


@st.cache_data
def build_tone_data_uri(frequency=660, duration_ms=180, volume=0.22):
    sample_rate = 22050
    frames = []
    total_samples = int(sample_rate * duration_ms / 1000)
    for index in range(total_samples):
        envelope = min(index / max(total_samples * 0.15, 1), 1.0)
        envelope *= max(0.0, 1.0 - index / max(total_samples, 1))
        sample = volume * envelope * math.sin(2 * math.pi * frequency * index / sample_rate)
        frames.append(struct.pack("<h", int(sample * 32767)))

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(b"".join(frames))

    encoded = base64.b64encode(buffer.getvalue()).decode()
    return f"data:audio/wav;base64,{encoded}"


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
        "language": "English",
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
if "screen" not in st.session_state:
    st.session_state.screen = "splash"

weather_data = fetch_live_weather()
if "ui_settings" not in st.session_state:
    st.session_state.ui_settings = default_ui_settings()
if "ui_draft" not in st.session_state:
    st.session_state.ui_draft = copy.deepcopy(st.session_state.ui_settings)

logo_path = "assets/logo.png"
logo_data_uri = load_logo_data_uri(logo_path)
intro_tone = build_tone_data_uri(660, 180)
home_tone = build_tone_data_uri(520, 220)

if st.session_state.screen == "splash":
    st.markdown(
        textwrap.dedent(
            f"""
            <style>
            .stApp {{ background:#000000 !important; }}
            [data-testid="collapsedControl"], [data-testid="stSidebar"], #MainMenu, footer, header {{ display:none !important; }}
            .block-container {{ padding-top:2rem; max-width:1100px; }}
            .intro-wrap {{ min-height:88vh; display:flex; align-items:center; justify-content:center; flex-direction:column; text-align:center; color:#f8fbff; }}
            .intro-title {{ font-family:'Sora', sans-serif; font-size:clamp(2.8rem, 8vw, 5.4rem); font-weight:800; letter-spacing:0.08em; margin:0; }}
            .intro-title span {{ display:inline-block; opacity:0; transform:translateY(18px) scale(0.96); animation:intro-letter 0.55s ease forwards; }}
            .intro-logo {{ width:110px; height:110px; margin:1.4rem auto 0.8rem; border-radius:26px; box-shadow:0 18px 60px rgba(255,255,255,0.12); opacity:0; animation:intro-logo 0.8s ease 1.9s forwards; }}
            .intro-copy {{ color:#b9c3d1; margin-top:0.8rem; opacity:0; animation:intro-logo 0.8s ease 2.1s forwards; }}
            .intro-loading {{ margin-top:1.4rem; color:#7ef0bb; font-weight:700; letter-spacing:0.18em; opacity:0; animation:intro-logo 0.8s ease 2.4s forwards; }}
            .intro-loading::after {{ content:""; animation:dots 1.4s steps(4,end) infinite; }}
            @keyframes intro-letter {{ to {{ opacity:1; transform:translateY(0) scale(1); }} }}
            @keyframes intro-logo {{ to {{ opacity:1; transform:translateY(0); }} }}
            @keyframes dots {{ 0% {{ content:""; }} 25% {{ content:"."; }} 50% {{ content:".."; }} 75% {{ content:"..."; }} 100% {{ content:""; }} }}
            </style>
            <audio autoplay><source src="{intro_tone}" type="audio/wav"></audio>
            <div class="intro-wrap">
                <h1 class="intro-title">{''.join(f'<span style="animation-delay:{0.12 * idx:.2f}s">{char}</span>' for idx, char in enumerate("RiceGenixAI"))}</h1>
                {'<img class="intro-logo" src="' + logo_data_uri + '" alt="RiceGenixAI logo">' if logo_data_uri else ''}
                <div class="intro-copy">Created by Aranyak Chakraborty</div>
                <div class="intro-loading">LOADING</div>
            </div>
            """
        ),
        unsafe_allow_html=True,
    )
    _, center_col, _ = st.columns([1.2, 1, 1.2])
    with center_col:
        progress_text = st.empty()
        progress_text.markdown(
            "<div style='text-align:center; color:#9fb3c6; font-weight:700; letter-spacing:0.08em; margin-bottom:0.5rem;'>Initializing RiceGenixAI...</div>",
            unsafe_allow_html=True,
        )
        progress_bar = st.progress(0, text="Loading experience...")
        for step in range(100):
            time.sleep(0.05)
            progress_bar.progress(step + 1, text=f"Loading experience... {step + 1}%")
        st.session_state.screen = "home"
        st.rerun()
    st.stop()

if st.session_state.screen == "home":
    st.markdown(
        textwrap.dedent(
            f"""
            <style>
            .home-wrap {{ min-height:85vh; display:flex; align-items:center; justify-content:center; }}
            .home-card {{ width:min(760px, 100%); padding:2rem; border-radius:32px; background:linear-gradient(180deg, rgba(10,19,31,0.92) 0%, rgba(17,33,50,0.86) 100%); border:1px solid rgba(255,255,255,0.1); box-shadow:0 30px 90px rgba(0,0,0,0.35); text-align:center; color:#f8fbff; }}
            .home-logo {{ width:120px; height:120px; border-radius:28px; margin-bottom:1rem; box-shadow:0 18px 52px rgba(34,211,238,0.22); }}
            .home-title {{ font-family:'Sora', sans-serif; font-size:clamp(2rem, 5vw, 3.2rem); margin:0 0 0.6rem 0; }}
            .home-copy {{ color:#a9b9c9; max-width:560px; margin:0 auto 1.4rem auto; line-height:1.7; }}
            </style>
            <audio autoplay><source src="{home_tone}" type="audio/wav"></audio>
            <div class="home-wrap">
                <div class="home-card">
                    {'<img class="home-logo" src="' + logo_data_uri + '" alt="RiceGenixAI logo">' if logo_data_uri else ''}
                    <div class="home-title">Welcome to RiceGenixAI</div>
                    <div class="home-copy">Smart yield prediction, disease detection, growth projection, weather support, and PDF reporting in one streamlined workspace.</div>
                </div>
            </div>
            """
        ),
        unsafe_allow_html=True,
    )
    st.session_state.ui_draft["language"] = st.selectbox(
        t("language"),
        LANGUAGE_OPTIONS,
        index=LANGUAGE_OPTIONS.index(st.session_state.ui_draft.get("language", "English")),
        format_func=lambda value: t(f"language_{value.lower()}") if value in LANGUAGE_OPTIONS else value,
    )
    if st.button("Start", use_container_width=True):
        st.session_state.ui_settings["language"] = st.session_state.ui_draft["language"]
        st.session_state.screen = "app"
        st.rerun()
    st.stop()

header_left, header_right = st.columns([7, 2])
with header_right:
    with st.expander(t("settings_title"), expanded=False):
        with st.form("ui_settings_form"):
            draft_theme = st.selectbox(
                t("theme_mode"),
                ["System Default", "Dark", "Light"],
                index=["System Default", "Dark", "Light"].index(st.session_state.ui_draft.get("theme_mode", st.session_state.ui_settings["theme_mode"])),
                format_func=lambda value: t(f"theme_{value.lower().replace(' ', '_')}") if value in ["System Default", "Dark", "Light"] else value,
            )
            draft_accent = st.selectbox(
                t("accent_palette"),
                ["Aurora", "Ocean", "Sunset", "Royal"],
                index=["Aurora", "Ocean", "Sunset", "Royal"].index(st.session_state.ui_draft.get("accent_palette", st.session_state.ui_settings["accent_palette"])),
            )
            draft_language = st.selectbox(
                t("language"),
                LANGUAGE_OPTIONS,
                index=LANGUAGE_OPTIONS.index(st.session_state.ui_draft.get("language", st.session_state.ui_settings["language"])),
                format_func=lambda value: t(f"language_{value.lower()}") if value in LANGUAGE_OPTIONS else value,
            )
            apply_settings = st.form_submit_button("Apply", use_container_width=True)
        if apply_settings:
            st.session_state.ui_draft["theme_mode"] = draft_theme
            st.session_state.ui_draft["accent_palette"] = draft_accent
            st.session_state.ui_draft["language"] = draft_language
            st.session_state.ui_settings["theme_mode"] = draft_theme
            st.session_state.ui_settings["accent_palette"] = draft_accent
            st.session_state.ui_settings["language"] = draft_language
            st.rerun()
        if st.button(t("reset_interface"), use_container_width=True):
            defaults = default_ui_settings()
            st.session_state.ui_settings = defaults
            st.session_state.ui_draft = copy.deepcopy(defaults)
            st.rerun()
        st.markdown('<div class="rg-settings-note">Use Apply to activate theme or language changes.</div>', unsafe_allow_html=True)

token_pack = get_theme_tokens(
    st.session_state.ui_settings["theme_mode"],
    st.session_state.ui_settings["accent_palette"],
)
inject_custom_theme(st.session_state.ui_settings, token_pack)

with header_left:
    weather_label = t("badge_weather_offline")
    if weather_data and weather_data.get("temperature") is not None:
        weather_label = t("live_weather", temperature=weather_data["temperature"])
    st.markdown(
        textwrap.dedent(
            f"""
            <div class="rg-shell">
                <div class="rg-hero">
                    <div style="display:flex; align-items:center; gap:1rem; flex-wrap:wrap;">
                        {'<img src="' + logo_data_uri + '" alt="RiceGenixAI logo" style="width:86px;height:86px;border-radius:22px;box-shadow:0 16px 44px rgba(0,0,0,0.18);">' if logo_data_uri else ''}
                        <div>
                            <div class="rg-badge">{t("precision_banner")}</div>
                            <div class="rg-title">RiceGenixAI</div>
                            <div class="rg-subtitle">{t("hero_subtitle")}</div>
                        </div>
                    </div>
                    <div style="display:flex; gap:0.7rem; flex-wrap:wrap; margin-top:0.6rem;">
                        <div class="rg-badge">{weather_label}</div>
                        <div class="rg-badge">{t("badge_smart_projection")}</div>
                        <div class="rg-badge">{t("badge_pdf_ai")}</div>
                    </div>
                </div>
            </div>
            """
        ),
        unsafe_allow_html=True,
    )

st.markdown(f'<div class="rg-floating-credit">{t("floating_credit")}</div>', unsafe_allow_html=True)

if model_dl is None:
    st.info(t("ai_unavailable_info"))

if not SKLEARN_AVAILABLE or yield_model is None:
    st.error(t("prediction_model_error"))
    st.stop()

main_col, side_col = st.columns([1.4, 0.8], gap="large")

with side_col:
    if weather_data:
        weather_items = [
            {"label": t("temperature_label"), "value": f"{weather_data['temperature']} C"},
            {"label": t("wind_label"), "value": f"{weather_data['windspeed']} km/h" if weather_data.get("windspeed") is not None else "N/A"},
            {"label": t("mode_label"), "value": st.session_state.ui_settings["theme_mode"]},
        ]
        render_metric_cards(weather_items)
    st.markdown(
        textwrap.dedent(
            f"""
            <div class="rg-card" style="margin-top:1rem;">
                <div class="rg-card-inner">
                    <div class="rg-section-title">{t('support_title')}</div>
                    <div class="rg-section-copy">{t('support_copy')}</div>
                </div>
            </div>
            """
        ),
        unsafe_allow_html=True,
    )
    if st.button(t("reset_inputs"), use_container_width=True):
        preserved = copy.deepcopy(st.session_state.ui_settings)
        preserved_draft = copy.deepcopy(st.session_state.ui_draft)
        preserved_screen = st.session_state.screen
        st.session_state.clear()
        st.session_state.ui_settings = preserved
        st.session_state.ui_draft = preserved_draft
        st.session_state.screen = preserved_screen
        st.session_state.result = None
        st.rerun()

with main_col:
    st.markdown(
        textwrap.dedent(
            f"""
            <div class="rg-card">
                <div class="rg-card-inner">
                    <div class="rg-section-title">{t('workspace_title')}</div>
                    <div class="rg-section-copy">{t('workspace_copy')}</div>
                </div>
            </div>
            """
        ),
        unsafe_allow_html=True,
    )

    with st.form("prediction_form"):
        upper_a, upper_b = st.columns(2, gap="large")
        with upper_a:
            crop_name = st.selectbox(t("select_rice_variety"), list(rice_data.keys()))
            height = st.number_input(t("enter_plant_height"), min_value=0.0, value=0.0)
            months_observed = st.number_input(t("months_observed"), min_value=0.5, max_value=12.0, value=1.0, step=0.5)
            gene_b = st.radio(t("disease_resistant"), ["Yes", "No"], horizontal=True, format_func=lambda value: t(value.lower()))
            gene_c = st.radio(t("drought_tolerant"), ["Yes", "No"], horizontal=True, format_func=lambda value: t(value.lower()))

        with upper_b:
            rain = st.number_input(t("annual_rainfall"), min_value=0.0, value=0.0)
            temp_unit = st.selectbox(t("temperature_unit"), ["Celsius", "Fahrenheit"], format_func=lambda value: t(value.lower()))
            temp_input = st.number_input(t("average_temperature"), value=0.0)
            soil_type = st.selectbox(t("soil_type"), ["Loamy", "Clay", "Sandy", "Alluvial", "Laterite"], format_func=lambda value: t(value.lower()))
            water_source = st.selectbox(t("irrigation_water_type"), ["Rainwater", "Groundwater", "Mixed"], format_func=lambda value: t(value.lower()))
            fertilizer_use = st.selectbox(t("fertilizer_usage"), ["Organic", "Chemical", "Mixed"], format_func=lambda value: t(value.lower()))

        st.markdown('<div class="rg-divider"></div>', unsafe_allow_html=True)
        soil_a, soil_b = st.columns(2, gap="large")
        with soil_a:
            manual_ph = st.checkbox(t("i_know_my_soil_ph"))
            ph_input = st.slider(t("enter_soil_ph"), 3.0, 9.0, 6.5, step=0.1)
        with soil_b:
            uploaded_file = st.file_uploader(t("upload_crop_image"), type=["jpg", "jpeg", "png"])
            camera_file = st.camera_input(t("use_phone_camera"))

        submitted = st.form_submit_button(t("predict_yield"), use_container_width=True)

preview_image, selected_image_source = resolve_image_input(uploaded_file, camera_file)
current_signature = build_input_signature(
    crop_name, height, months_observed, gene_b, gene_c, rain, temp_unit, temp_input,
    soil_type, water_source, fertilizer_use, manual_ph, ph_input, preview_image is not None
)

if submitted:
    try:
        st.session_state.result = None
        rain_val = float(rain)
        temp_val = float(temp_input)
        height_val = float(height)

        expected_height = rice_data[crop_name]["height"]
        growth_metrics = project_growth_metrics(crop_name, height_val, months_observed)
        projected_final_height = growth_metrics["projected_final_height"]
        expected_height_now = growth_metrics["expected_height_now"]
        maturity_months = growth_metrics["maturity_months"]
        g4 = 1 if height_val >= expected_height_now * 1.05 or projected_final_height >= expected_height * 1.05 else 0
        g1 = 1 if projected_final_height >= expected_height else 0
        g2 = 1 if gene_b == "Yes" else 0
        g3 = 1 if gene_c == "Yes" else 0

        disease_name = "Not Checked"
        image_for_report = None
        if preview_image is not None:
            try:
                image_for_prediction = preview_image.copy()
                disease_name = predict(model_dl, image_for_prediction)
                image_for_report = image_for_prediction.copy()
            except Exception:
                disease_name = "Model Error"

        if temp_unit == "Fahrenheit":
            temp_val = (temp_val - 32) * 5 / 9

        ph = estimate_ph(soil_type, water_source, fertilizer_use, manual_ph, ph_input)
        base_pred = yield_model.predict([[g1, g2, g3, g4, rain_val, temp_val, ph]])[0]
        disease, water = crop_health(g2, rain_val, temp_val)
        final_pred = float(base_pred)

        height_status = growth_metrics.get("height_status", "Growth status unavailable.")
        height_flag = height_status

        if height_val >= expected_height_now * 1.05:
            final_pred *= 1.06
        elif height_val >= expected_height_now * 0.95:
            final_pred *= 1.02
        elif height_val >= expected_height_now * 0.85:
            final_pred *= 0.96
        elif height_val >= expected_height_now * 0.70:
            final_pred *= 0.90
        else:
            final_pred *= 0.82

        if projected_final_height < expected_height:
            deficit = (expected_height - projected_final_height) / expected_height
            final_pred -= 0.9 * min(deficit, 1.0)
            height_flag += " Projected final height is below the crop's average target."
        elif projected_final_height > expected_height * 1.08:
            final_pred -= 0.3
            height_flag += " Projected height is above the usual target, which may signal nutrient imbalance."
        else:
            height_flag += " Projection is within the flexible growth band."

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
    except Exception as exc:
        st.error(f"Prediction error: {exc}")
        st.stop()

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
            {"label": "Growth Status", "value": res["height_flag"]},
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
        st.json({"Gene_A": res["genes"][0], "Gene_B": res["genes"][1], "Gene_C": res["genes"][2], "Gene_D": res["genes"][3]})

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
            st.write(t("very_low_rainfall"))
        elif res["rain"] < 200:
            st.write(t("moderately_low_rainfall"))
        elif res["rain"] > 400:
            st.write(t("excess_rainfall"))

        if res["temp"] > 40:
            st.write(t("high_temperature_stress"))
        elif res["temp"] < 20:
            st.write(t("low_temperature_slow"))

        st.write(t("use_high_yield_seeds"))
        st.write(t("apply_balanced_npk"))
        st.write(t("monitor_weekly"))
        st.write(t("use_proper_spacing"))

    if preview_image is not None:
        st.markdown("### Uploaded Image")
        st.image(preview_image, caption="Uploaded Image", use_container_width=True)

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
        label=t("download_pdf_report"),
        data=pdf_bytes,
        file_name="RiceGenix_Report.pdf",
        mime="application/pdf",
        key="pdf_download",
        use_container_width=True,
    )
    st.markdown(
        f'<a href="data:application/pdf;base64,{pdf_b64}" download="RiceGenix_Report.pdf">{t("mobile_download_fails")}</a>',
        unsafe_allow_html=True,
    )
