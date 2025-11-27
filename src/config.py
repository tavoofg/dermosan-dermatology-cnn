"""
Configuration of the dermatological diagnostic system
"""

import os
from typing import Dict, List, Tuple

# ── Model Configuration ──────────────────────────────────────────────────
MODEL_PATH = "models/best_resnet152.h5"  # Corrected path
MODEL_PATH_FALLBACK = "Trained_Model/best_resnet152.h5"  # Alternative path
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# ── Dermatological Disease Classes ─────────────────────────────────────────
DISEASE_CLASSES = {
    0: "Eczema",
    1: "Warts Molluscum and Viral Infections", 
    2: "Melanoma",
    3: "Atopic Dermatitis",
    4: "Basal Cell Carcinoma (BCC)",
    5: "Melanocytic Nevi (NV)",
    6: "Benign Keratosis-like Lesions (BKL)",
    7: "Psoriasis Lichen Planus",
    8: "Seborrheic Keratoses",
    9: "Tinea Ringworm Candidiasis"
}

# ── Detailed Medical Information ──────────────────────────────────────────────
DISEASE_INFO = {
    "Eczema": {
        "description": "Chronic inflammation of the skin characterized by red rashes, itching, and scaling.",
        "severity": "Mild to Moderate",
        "treatment": "Moisturizers, topical corticosteroids, antihistamines",
        "color": "#E74C3C"  # Medical Red
    },
    "Warts Molluscum and Viral Infections": {
        "description": "Viral skin infections causing small bumps or warts.",
        "severity": "Mild",
        "treatment": "Crioterapia, medicamentos tópicos, observación",
        "color": "#1ABC9C"  # Medical Teal
    },
    "Melanoma": {
        "description": "Most dangerous type of skin cancer that develops in melanocytes.",
        "severity": "Severe - Requires immediate attention",
        "treatment": "Surgery, immunotherapy, targeted therapy",
        "color": "#8E44AD"  # Medical Purple (critical)
    },
    "Atopic Dermatitis": {
        "description": "Common chronic eczema in children, characterized by dry and itchy skin.",
        "severity": "Mild to Moderate",
        "treatment": "Moisturizers, corticosteroids, immunomodulators",
        "color": "#F39C12"  # Medical Orange
    },
    "Basal Cell Carcinoma (BCC)": {
        "description": "Most common type of skin cancer, slow growing and rarely metastatic.",
        "severity": "Moderate",
        "treatment": "Surgery, cryotherapy, topical medications",
        "color": "#C0392B"  # Medical Dark Red
    },
    "Melanocytic Nevi (NV)": {
        "description": "Common benign moles, generally do not require treatment.",
        "severity": "Benign",
        "treatment": "Observation, biopsy if changes occur",
        "color": "#27AE60"  # Medical Green
    },
    "Benign Keratosis-like Lesions (BKL)": {
        "description": "Benign skin lesions, including seborrheic keratosis and similar lesions.",
        "severity": "Benign",
        "treatment": "Observation, cosmetic removal if desired",
        "color": "#3498DB"  # Medical Blue
    },
    "Psoriasis Lichen Planus": {
        "description": "Chronic inflammatory skin diseases with scaly plaques.",
        "severity": "Moderate",
        "treatment": "Corticosteroids, immunosuppressants, phototherapy",
        "color": "#9B59B6"  # Medical Light Purple
    },
    "Seborrheic Keratoses": {
        "description": "Benign skin growths, common in older adults.",
        "severity": "Benign",
        "treatment": "Observation, cosmetic removal",
        "color": "#16A085"  # Medical Dark Green
    },
    "Tinea Ringworm Candidiasis": {
        "description": "Fungal skin infections causing circular rashes or irritation.",
        "severity": "Mild to Moderate",
        "treatment": "Topical or oral antifungals",
        "color": "#E67E22"  # Medical Dark Orange
    }
}

# ── Configuración de la aplicación ────────────────────────────────────────────
APP_CONFIG = {
    "title": " Dermosan - Dermatological Diagnosis System",
    "subtitle": "DERMOSAN – UNDC 2025",
    "description": "Automated system for diagnosing dermatological diseases using Deep Learning"
}

# ── Confidence Configuration ────────────────────────────────────────────────
CONFIDENCE_THRESHOLDS = {
    "high": 0.8,      # High confidence
    "medium": 0.6,    # Medium confidence
    "low": 0.4        # Low confidence
}
