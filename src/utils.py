"""
Utilities for the dermatological diagnostic system
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List
import base64
import io

from src.config import DISEASE_INFO, APP_CONFIG
from src.styles import apply_custom_styling, create_metric_card, create_status_indicator

def set_page_config():
    """Configures the Streamlit page."""
    st.set_page_config(
        page_title=APP_CONFIG["title"],
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    # Apply custom styles
    apply_custom_styling()

def display_header():
    """Displays the application header with enhanced information."""
    st.markdown("""
    <div style="background: linear-gradient(135deg, #2E5BBA, #4A90B8); 
                padding: 2rem; border-radius: 15px; margin-bottom: 2rem; 
                box-shadow: 0 8px 24px rgba(46, 91, 186, 0.15);">
        <h1 style="color: white; text-align: center; margin: 0; font-size: 2.5rem; font-weight: 600;">
             Dermosan - Dermatological Diagnostic System
        </h1>
        <p style="color: white; text-align: center; margin: 0.5rem 0 0 0; font-size: 1.2rem; opacity: 0.95;">
           DERMOSAN Project – UNDC 2025
        </p>
        <div style="text-align: center; margin-top: 1.5rem;">
            <span style="background: rgba(255,255,255,0.2); color: white; padding: 0.5rem 1.5rem; 
                         border-radius: 25px; font-size: 0.95rem; border: 1px solid rgba(255,255,255,0.3);">
                 AI ResNet152 |  95% Accuracy |  10 Diseases
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

def display_sidebar_info():
    """Displays enhanced information in the sidebar."""
    with st.sidebar:
        st.markdown("### System Information")
        
        # Model metrics in colorful boxes
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #27AE60, #2ECC71); color: white; 
                        padding: 1rem; border-radius: 12px; text-align: center; margin-bottom: 0.5rem;
                        box-shadow: 0 4px 12px rgba(39, 174, 96, 0.2);">
                <h3 style="margin: 0; font-size: 1.5rem; font-weight: 600;">95%</h3>
                <p style="margin: 0; font-size: 0.8rem; opacity: 0.9;">Accuracy</p>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #3498DB, #5DADE2); color: white; 
                        padding: 1rem; border-radius: 12px; text-align: center; margin-bottom: 0.5rem;
                        box-shadow: 0 4px 12px rgba(52, 152, 219, 0.2);">
                <h3 style="margin: 0; font-size: 1.5rem; font-weight: 600;">10</h3>
                <p style="margin: 0; font-size: 0.8rem; opacity: 0.9;">Diseases</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("""
        **Model:** ResNet152 with Transfer Learning  
        **Dataset:** +27,000 images  
        **Input:** 224x224 RGB  
        **Average Time:** < 3 seconds  
        """)
        
        # Add system status
        st.markdown("### 🔋 System Status")
        st.markdown(create_status_indicator("active", "Operating System"), unsafe_allow_html=True)
        st.markdown(create_status_indicator("active", "Model Loaded"), unsafe_allow_html=True)
        st.markdown(create_status_indicator("active", "AI Ready for Diagnosis"), unsafe_allow_html=True)
        
        st.markdown("### Dataset Statistics")
    
        dataset_stats = {
            "Eczema": 1677,
            "Melanoma": 3140,
            "Atopic Dermatitis": 1257,
            "Basal Cell Carcinoma (BCC)": 3323,
            "Melanocytic Nevi (NV)": 7970,
            "Benign Keratosis-like Lesions (BKL)": 2624,
            "Psoriasis / Lichen Planus & related diseases": 2055,
            "Seborrheic Keratoses & other benign tumors": 1847,
            "Tinea / Candidiasis & other fungal infections": 1702,
            "Warts / Molluscum & other viral infections": 2103
        }

        
        df_stats = pd.DataFrame(
            list(dataset_stats.items()), 
            columns=['Disease', 'Images']
        )
        st.dataframe(df_stats, hide_index=True, width='stretch')

def create_confidence_gauge(confidence: float) -> go.Figure:
    """
    Creates a gauge chart to display confidence with a medical design.
    
    Args:
        confidence: Confidence value (0-1)
        
    Returns:
        Plotly figure
    """
    confidence_percentage = confidence * 100
    
    # Determine color based on confidence
    if confidence >= 0.8:
        color = "#27AE60"  # Medical green
        bar_color = "rgba(39, 174, 96, 0.9)"
    elif confidence >= 0.6:
        color = "#F39C12"  # Medical orange
        bar_color = "rgba(243, 156, 18, 0.9)"
    else:
        color = "#E74C3C"  # Medical red
        bar_color = "rgba(231, 76, 60, 0.9)"
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence_percentage,
        title={'text': "Nivel de Confianza (%)", 'font': {'size': 18, 'color': '#2E5BBA'}},
        domain={'x': [0, 1], 'y': [0, 1]},
        number={'font': {'size': 40, 'color': '#2E5BBA', 'family': 'Arial Black'}},
        gauge={
            'axis': {
                'range': [None, 100], 
                'tickwidth': 2, 
                'tickcolor': "#34495E",
                'tickfont': {'size': 14, 'color': '#34495E'}
            },
            'bar': {
                'color': bar_color, 
                'thickness': 0.8,
                'line': {'color': '#2E5BBA', 'width': 3}
            },
            'bgcolor': "rgba(240, 248, 255, 0.3)",
            'borderwidth': 4,
            'bordercolor': "#2E5BBA",
            'steps': [
                {'range': [0, 40], 'color': "rgba(231, 76, 60, 0.2)", 'name': 'Bajo'},
                {'range': [40, 60], 'color': "rgba(243, 156, 18, 0.2)", 'name': 'Medio'},
                {'range': [60, 80], 'color': "rgba(52, 152, 219, 0.2)", 'name': 'Bueno'},
                {'range': [80, 100], 'color': "rgba(39, 174, 96, 0.2)", 'name': 'Excelente'}
            ],
            'threshold': {
                'line': {'color': "#E74C3C", 'width': 5},
                'thickness': 0.9,
                'value': 85
            }
        }
    ))
    
    fig.update_layout(
        height=350, 
        margin=dict(l=20, r=20, t=50, b=20),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(
            family="Arial, sans-serif",
            size=14,
            color="#34495E"
        )
    )
    return fig

def create_probability_chart(probabilities: Dict[str, float]) -> go.Figure:
    """
    Creates a sleek horizontal bar chart with probabilities.
    
    Args:
        probabilities: Dictionary with class probabilities
        
    Returns:
        Plotly figure
    """
    # Prepare data
    diseases = list(probabilities.keys())
    probs = [probabilities[disease] * 100 for disease in diseases]
    
    # Check if we have data
    if not probs:
        fig = go.Figure()
        fig.update_layout(title="No data to display")
        return fig
    
    # Get colors and sort by probability
    disease_data = [(disease, prob, DISEASE_INFO.get(disease, {}).get("color", "#636EFA")) 
                    for disease, prob in zip(diseases, probs)]
    disease_data.sort(key=lambda x: x[1], reverse=True)
    
    diseases_sorted, probs_sorted, colors_sorted = zip(*disease_data)
    
    fig = go.Figure(data=[
        go.Bar(
            x=list(probs_sorted),
            y=list(diseases_sorted),
            orientation='h',
            marker=dict(
                color=list(colors_sorted),
                line=dict(color='#2E5BBA', width=2),
                opacity=0.8
            ),
            text=[f"{prob:.1f}%" for prob in probs_sorted],
            textposition='outside',
            textfont=dict(size=12, color='#2E5BBA', family='Arial Bold')
        )
    ])
    
    # Calcular rango del eje X
    max_prob = max(probs_sorted) if probs_sorted else 100
    x_range = [0, max_prob * 1.15]
    
    fig.update_layout(
        title=dict(
            text="🔬 Probability Analysis by Disease",
            font=dict(size=18, color='#2E5BBA', family="Arial Bold"),
            x=0.5
        ),
        xaxis_title="Probability (%)",
        yaxis_title="Detected Disease",
        height=500,
        margin=dict(l=150, r=50, t=80, b=50),
        xaxis=dict(
            range=x_range,
            gridcolor='rgba(46, 91, 186, 0.1)',
            showgrid=True
        ),
        yaxis=dict(
            gridcolor='rgba(46, 91, 186, 0.1)',
            showgrid=True
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(
            family="Arial, sans-serif",
            size=12,
            color='#34495E'
        ),
        xaxis_title_font=dict(size=14, color='#2E5BBA'),
        yaxis_title_font=dict(size=14, color='#2E5BBA'),
        showlegend=False
    )
    return fig

def create_compact_probability_chart(probabilities: Dict[str, float]) -> go.Figure:
    """
    Creates a compact horizontal bar chart for the main panel.
    
    Args:
        probabilities: Dictionary with class probabilities
        
    Returns:
        Compact Plotly figure
    """
    # Prepare data - show only top 5
    diseases = list(probabilities.keys())
    probs = [probabilities[disease] * 100 for disease in diseases]
    
    if not probs:
        fig = go.Figure()
        fig.update_layout(title="No data to display")
        return fig
    
    # Get colors and sort by probability (top 5)
    disease_data = [(disease, prob, DISEASE_INFO.get(disease, {}).get("color", "#636EFA")) 
                    for disease, prob in zip(diseases, probs)]
    disease_data.sort(key=lambda x: x[1], reverse=True)
    disease_data = disease_data[:5]  # Only top 5
    
    diseases_sorted, probs_sorted, colors_sorted = zip(*disease_data)
    
    # Shorten disease names for compact view
    diseases_short = []
    for disease in diseases_sorted:
        if len(disease) > 20:
            diseases_short.append(disease[:17] + "...")
        else:
            diseases_short.append(disease)
    
    fig = go.Figure(data=[
        go.Bar(
            x=list(probs_sorted),
            y=list(diseases_short),
            orientation='h',
            marker=dict(
                color=list(colors_sorted),
                line=dict(color='#2E5BBA', width=1),
                opacity=0.8
            ),
            text=[f"{prob:.1f}%" for prob in probs_sorted],
            textposition='outside',
            textfont=dict(size=10, color='#2E5BBA', family='Arial Bold')
        )
    ])
    
    max_prob = max(probs_sorted) if probs_sorted else 100
    x_range = [0, max_prob * 1.2]
    
    fig.update_layout(
        height=300,  # More compact
        margin=dict(l=100, r=20, t=20, b=20),  # Reduced margins
        xaxis=dict(
            range=x_range,
            gridcolor='rgba(46, 91, 186, 0.1)',
            showgrid=True,
            showticklabels=True
        ),
        yaxis=dict(
            gridcolor='rgba(46, 91, 186, 0.1)',
            showgrid=False,
            showticklabels=True
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(
            family="Arial, sans-serif",
            size=10,
            color='#34495E'
        ),
        showlegend=False
    )
    return fig

def display_results_container(content_func):
    """
    Wrapper para mostrar contenido en un contenedor con diseño mejorado.
    
    Args:
        content_func: Function that generates the content
    """
    st.markdown("""
    <div class="result-container">
    """, unsafe_allow_html=True)
    
    content_func()
    
    st.markdown("""
    </div>
    """, unsafe_allow_html=True)

def display_confidence_level(confidence: float):
    """
    Displays the confidence level with colors and medical style.
    
    Args:
        confidence: Confidence value (0-1)
    """
    confidence_percentage = confidence * 100
    
    if confidence >= 0.8:
        css_class = "confidence-high"
        level = "High"
        icon = "🟢"
    elif confidence >= 0.6:
        css_class = "confidence-medium" 
        level = "Medium"
        icon = "🟡"
    else:
        css_class = "confidence-low"
        level = "Low"
        icon = "🔴"
    
    st.markdown(f"""
    <div style="text-align: center; padding: 1rem; background: white; border-radius: 10px; 
                border: 2px solid #E8F4FD; margin: 1rem 0;">
        <h3 style="margin: 0; color: #2E5BBA;">Confidence Level</h3>
        <div style="margin: 1rem 0;">
            <span style="font-size: 2rem;">{icon}</span>
            <span class="{css_class}" style="font-size: 2.5rem; margin-left: 0.5rem;">
                {confidence_percentage:.1f}%
            </span>
        </div>
        <p style="margin: 0; color: #7F8C8D;">Confidence: {level}</p>
    </div>
    """, unsafe_allow_html=True)

def display_disease_info(disease_name: str):
    """
    Displays detailed information about a disease.
    
    Args:
        disease_name: Name of the disease
    """
    if disease_name in DISEASE_INFO:
        info = DISEASE_INFO[disease_name]
        
        # Determine emoji based on severity
        severity_emoji = {
            "Benign": "✅",
            "Mild": "💛", 
            "Mild to Moderate": "🟡",
            "Moderate": "🟠",
            "Severe - Requires immediate attention": "🚨"
        }
        
        emoji = severity_emoji.get(info["severity"], "ℹ️")
        
        st.markdown(f"""
        <div style="background-color: {info['color']}15; 
                    border-left: 4px solid {info['color']}; 
                    padding: 1rem; margin: 1rem 0; border-radius: 5px;">
            <h4 style="margin: 0; color: {info['color']};">
                {emoji} {disease_name}
            </h4>
            <p><strong>Description:</strong> {info['description']}</p>
            <p><strong>Severity:</strong> {info['severity']}</p>
            <p><strong>Typical Treatment:</strong> {info['treatment']}</p>
        </div>
        """, unsafe_allow_html=True)

def display_quality_analysis(quality_result: Dict):
    """
    Displays the image quality analysis with enhanced medical validation.
    
    Args:
        quality_result: Result of the quality analysis
    """
    # 1. CRITICAL VALIDATION: Check if it is a valid medical image
    if not quality_result.get("is_medical_image", True):
        error_type = quality_result.get("validation_error", "unknown")
        
        # Show critical error based on type
        if error_type == "text_document":
            st.markdown("""
            <div style="background: #FFEBEE; 
                        border: 3px solid #F44336; 
                        padding: 2rem; 
                        border-radius: 15px; 
                        text-align: center;
                        margin: 1rem 0;
                        box-shadow: 0 4px 8px rgba(244,67,54,0.3);">
                <div style="font-size: 4rem; color: #F44336; margin-bottom: 1rem;">🚫</div>
                <h2 style="color: #C62828; margin: 0 0 1rem 0;">
                    DOCUMENT DETECTED
                </h2>
                <p style="color: #D32F2F; font-size: 1.3rem; margin: 0; line-height: 1.5;">
                    <strong>This image contains text/document, not a medical photograph.</strong>
                </p>
            </div>
            """, unsafe_allow_html=True)
            
        elif error_type == "screenshot":
            st.markdown("""
            <div style="background: #FFEBEE; 
                        border: 3px solid #F44336; 
                        padding: 2rem; 
                        border-radius: 15px; 
                        text-align: center;
                        margin: 1rem 0;
                        box-shadow: 0 4px 8px rgba(244,67,54,0.3);">
                <div style="font-size: 4rem; color: #F44336; margin-bottom: 1rem;">📱</div>
                <h2 style="color: #C62828; margin: 0 0 1rem 0;">
                    SCREENSHOT DETECTED
                </h2>
                <p style="color: #D32F2F; font-size: 1.3rem; margin: 0; line-height: 1.5;">
                    <strong>Screenshots are not allowed.</strong><br>
                    Use a camera to photograph the skin directly.
                </p>
            </div>
            """, unsafe_allow_html=True)
            
        elif error_type == "not_medical":
            st.markdown("""
            <div style="background: #FFEBEE; 
                        border: 3px solid #F44336; 
                        padding: 2rem; 
                        border-radius: 15px; 
                        text-align: center;
                        margin: 1rem 0;
                        box-shadow: 0 4px 8px rgba(244,67,54,0.3);">
                <div style="font-size: 4rem; color: #F44336; margin-bottom: 1rem;">🏥</div>
                <h2 style="color: #C62828; margin: 0 0 1rem 0;">
                    NOT MEDICAL IMAGE
                </h2>
                <p style="color: #D32F2F; font-size: 1.3rem; margin: 0; line-height: 1.5;">
                    <strong>This image does not appear to be of skin or a skin lesion.</strong><br>
                    System designed only for dermatological images.
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        # Show specific issues
        if quality_result.get("issues"):
            st.markdown("### ❌ **Detected Issues:**")
            for issue in quality_result["issues"]:
                st.markdown(f"""
                <div style="background: #FFCDD2; 
                            border-left: 4px solid #F44336; 
                            padding: 1rem; 
                            margin: 0.5rem 0; 
                            border-radius: 4px;">
                    <strong style="color: #B71C1C;">{issue}</strong>
                </div>
                """, unsafe_allow_html=True)
        
        # Instructions for valid image
        st.markdown("""
        <div style="background: #E8F5E8; 
                    border-left: 4px solid #4CAF50; 
                    padding: 2rem; 
                    margin: 1.5rem 0; 
                    border-radius: 8px;">
            <h3 style="color: #2E7D32; margin: 0 0 1rem 0;">
                📸 How to take a correct dermatological photograph:
            </h3>
            <div style="color: #388E3C;">
                <p><strong>✅ Use direct photography with a camera</strong></p>
                <p><strong>✅ Focus the skin or lesion clearly</strong></p>
                <p><strong>✅ Good natural lighting</strong></p>
                <p><strong>✅ Appropriate distance (not too close/far)</strong></p>
                <p><strong>✅ Neutral background without distractions</strong></p>
                <br>
                <p><strong>❌ DO NOT use screenshots</strong></p>
                <p><strong>❌ DO NOT upload documents or texts</strong></p>
                <p><strong>❌ DO NOT use images from the internet</strong></p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        return # Exit without showing quality analysis
    
    # 2. If it is a valid medical image, show normal analysis
    score = quality_result["quality_score"]
    
    # Determine color and message
    if score >= 75:
        color = "#4CAF50"
        bg_color = "#E8F5E8"
        icon = "✅"
        status = "OPTIMAL"
        message = "Good quality image for diagnosis"
    elif score >= 50:
        color = "#FF9800"
        bg_color = "#FFF3E0"
        icon = "⚠️"
        status = "ACCEPTABLE"
        message = "Acceptable quality, but could be improved"
    else:
        color = "#F44336"
        bg_color = "#FFEBEE"
        icon = "❌"
        status = "INSUFFICIENT"
        message = "Insufficient quality for reliable diagnosis"
    
    # Main quality card
    st.markdown(f"""
    <div style="background: {bg_color}; 
                border-left: 4px solid {color}; 
                padding: 1.5rem; 
                margin: 1rem 0; 
                border-radius: 8px; 
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
        <div style="display: flex; align-items: center; margin-bottom: 1rem;">
            <div style="font-size: 1.5rem; margin-right: 0.5rem;">{icon}</div>
            <h3 style="color: {color}; margin: 0; font-size: 1.2rem;">
                CALIDAD DE IMAGEN: {status}
            </h3>
        </div>
        <p style="margin: 0.5rem 0; font-size: 1rem; color: #333;">
            <strong>{message}</strong>
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Technical metrics in small cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div style="background: white; 
                    border: 1px solid #E0E0E0; 
                    padding: 1rem; 
                    border-radius: 8px; 
                    text-align: center;
                    box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
            <h4 style="color: {color}; margin: 0; font-size: 1.5rem;">{score}</h4>
            <p style="margin: 0.2rem 0 0 0; font-size: 0.9rem; color: #666;">
                Score / 100
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        resolution = quality_result.get('resolution', 'N/A')
        st.markdown(f"""
        <div style="background: white; 
                    border: 1px solid #E0E0E0; 
                    padding: 1rem; 
                    border-radius: 8px; 
                    text-align: center;
                    box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
            <h4 style="color: #2196F3; margin: 0; font-size: 1.2rem;">{resolution}</h4>
            <p style="margin: 0.2rem 0 0 0; font-size: 0.9rem; color: #666;">
                Resolution
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        # Suitability indicator
        suitable = quality_result.get('is_suitable', False)
        suitable_text = "SUITABLE" if suitable else "NOT SUITABLE"
        suitable_color = "#4CAF50" if suitable else "#F44336"
        
        st.markdown(f"""
        <div style="background: white; 
                    border: 1px solid #E0E0E0; 
                    padding: 1rem; 
                    border-radius: 8px; 
                    text-align: center;
                    box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
            <h4 style="color: {suitable_color}; margin: 0; font-size: 1.2rem;">{suitable_text}</h4>
            <p style="margin: 0.2rem 0 0 0; font-size: 0.9rem; color: #666;">
                For diagnosis
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Show detected issues if any
    if quality_result.get("issues"):
        st.markdown("---")
        st.markdown("### 🔍 **Detailed Technical Analysis**")
        
        issues_html = ""
        for issue in quality_result["issues"]:
            issues_html += f"""
            <div style="background: #FFF3E0; 
                        border-left: 3px solid #FF9800; 
                        padding: 0.8rem; 
                        margin: 0.5rem 0; 
                        border-radius: 4px;">
                <span style="color: #F57C00;">⚠️</span> 
                <strong style="color: #E65100;">{issue}</strong>
            </div>
            """
        
        st.markdown(issues_html, unsafe_allow_html=True)
        
        # Recommendations for improvement
        st.markdown("""
        <div style="background: #E3F2FD; 
                    border-left: 3px solid #2196F3; 
                    padding: 1rem; 
                    margin: 1rem 0; 
                    border-radius: 4px;">
            <h4 style="color: #1976D2; margin: 0 0 0.5rem 0;">
                💡 Recommendations for Improving the Image:
            </h4>
            <ul style="margin: 0; color: #1565C0;">
                <li>Use natural lighting or uniform white light</li>
                <li>Keep the camera steady to avoid blurriness</li>
                <li>Ensure the lesion is well focused</li>
                <li>Use a minimum resolution of 224x224 pixels</li>
                <li>Avoid shadows and reflections in the image</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

def display_medical_recommendations(recommendations: Dict):
    """
    Displays medical recommendations.
    
    Args:
        recommendations: Dictionary with recommendations
    """
    urgency = recommendations["urgency"]
    
    # Determine color based on urgency
    if "URGENT" in urgency.upper():
        color = "#FF4757"
        icon = "🚨"
    elif "PRIORITY" in urgency.upper():
        color = "#FFA726"
        icon = "⚠️"
    else:
        color = "#42A5F5"
        icon = "ℹ️"
    
    st.markdown(f"""
    <div style="background-color: {color}15; 
                border-left: 4px solid {color}; 
                padding: 1.5rem; margin: 1rem 0; border-radius: 5px;">
        <h4 style="color: {color}; margin: 0;">
            {icon} Medical Recommendations
        </h4>
        <p><strong>Urgency:</strong> {urgency}</p>
        <p><strong>Recommended Action:</strong> {recommendations['recommended_action']}</p>
        <p><strong>Follow-up:</strong> {recommendations['follow_up']}</p>
    </div>
    """, unsafe_allow_html=True)

def export_diagnosis_report(image, prediction_result: Dict, quality_result: Dict, recommendations: Dict) -> str:
    """
    Generates a diagnosis report in text format.
    
    Args:
        image: Analyzed image
        prediction_result: Prediction result
        quality_result: Quality analysis result
        recommendations: Medical recommendations
        
    Returns:
        String with the report
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    report = f"""
    AUTOMATED DERMATOLOGICAL DIAGNOSIS REPORT
    =================================================
    
    Date and Time: {timestamp}
    System: Dermosan v1.0.0
    Clinic: DERMOSAN – UNDC 2025
    
    IMAGE ANALYSIS:
    ---------------
    Image Quality: {quality_result['quality_score']}/100
    Resolution: {quality_result.get('resolution', 'N/A')}
    Suitable for Diagnosis: {'Yes' if quality_result['is_suitable'] else 'No'}
    
    DIAGNOSIS RESULTS:
    -----------------
    Primary Diagnosis: {prediction_result['predicted_class']}
    Confidence: {prediction_result['confidence_percentage']} ({prediction_result['confidence_level']})
    
    TOP 3 DIFFERENTIAL DIAGNOSES:
    """
    
    for i, pred in enumerate(prediction_result['top_3_predictions'], 1):
        report += f"\n    {i}. {pred['disease']} - {pred['percentage']}"
    
    report += f"""
    
    CLINICAL RECOMMENDATIONS:
    ------------------------
    Urgency: {recommendations['urgency']}
    Recommended Action: {recommendations['recommended_action']}
    Follow-up: {recommendations['follow_up']}
    
    IMPORTANT:
    ----------
    This diagnosis is generated by artificial intelligence and must be
    validated by a qualified medical professional. It does not replace the
    clinical judgment or in-person evaluation.
    
    """
    
    return report

def display_model_metrics():
    """Displays detailed model metrics."""
    st.markdown("### 📈 ResNet152 Model Metrics")
    
    # MMetrics in columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #4CAF50, #45a049); color: white; 
                    padding: 1.5rem; border-radius: 10px; text-align: center;">
            <h2 style="margin: 0; font-size: 2rem;">95%</h2>
            <p style="margin: 0.5rem 0 0 0;">Overall Accuracy</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #2196F3, #1976D2); color: white; 
                    padding: 1.5rem; border-radius: 10px; text-align: center;">
            <h2 style="margin: 0; font-size: 2rem;">30K+</h2>
            <p style="margin: 0.5rem 0 0 0;">Training Images</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #FF9800, #F57C00); color: white; 
                    padding: 1.5rem; border-radius: 10px; text-align: center;">
            <h2 style="margin: 0; font-size: 2rem;">ResNet152</h2>
            <p style="margin: 0.5rem 0 0 0;">Advanced Architecture</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Technical Information
    st.markdown("#### 🔧 Technical Details")
    tech_info = {
        "Architecture": "ResNet152 + Transfer Learning",
        "Trainable Layers": "Last 50 layers",
        "Optimizer": "Adam with adaptive learning rate",
        "Loss Function": "Sparse Categorical Crossentropy",
        "Augmentation": "Rotation, zoom, horizontal flip",
        "Balancing": "Automatic class weights",
        "Validation": "Split 80/10/10 (train/val/test)",
        "Training Time": "~6 hours on GPU"
    }
    
    tech_df = pd.DataFrame(list(tech_info.items()), columns=['Parameter', 'Value'])
    st.dataframe(tech_df, hide_index=True, width='stretch')

def create_download_link(content: str, filename: str, link_text: str) -> str:
    """
    Creates a download link for text content.
    
    Args:
        content: Content to download
        filename: File name
        link_text: Link text
        
    Returns:
        HTML of the download link
    """
    b64 = base64.b64encode(content.encode()).decode()
    href = f'<a href="data:text/plain;base64,{b64}" download="{filename}">{link_text}</a>'
    return href

def create_risk_assessment_chart(predicted_disease: str, confidence: float) -> go.Figure:
    """
    Creates a medical risk assessment chart.
    
    Args:
        predicted_disease: Predicted disease
        confidence: Confidence level
        
    Returns:
        Plotly figure with risk assessment
    """
    # Get disease information
    disease_info = DISEASE_INFO.get(predicted_disease, {})
    severity_level = disease_info.get("severity", "Moderate")
    
    # Define risk levels
    risk_levels = {
        "Low": {"value": 25, "color": "#27AE60", "description": "Regular monitoring"},
        "Moderate": {"value": 50, "color": "#F39C12", "description": "Dermatological consultation"},
        "High": {"value": 75, "color": "#E67E22", "description": "Immediate attention"},
        "Critical": {"value": 100, "color": "#E74C3C", "description": "Medical emergency"}
    }
    
    # Determine risk based on disease and confidence
    if predicted_disease in ["Melanoma", "Basal Cell Carcinoma (BCC)"]:
        if confidence >= 0.8:
            current_risk = "High"
        else:
            current_risk = "Moderate"
    elif predicted_disease in ["Atopic Dermatitis", "Eczema", "Psoriasis pictures Lichen Planus and related diseases"]:
        current_risk = "Moderate" if confidence >= 0.7 else "Low"
    else:
        current_risk = "Low" if confidence >= 0.8 else "Moderate"
    
    # Create the chart
    categories = list(risk_levels.keys())
    values = [risk_levels[cat]["value"] for cat in categories]
    colors = [risk_levels[cat]["color"] for cat in categories]
    
    # Highlight the current level
    opacity = [0.9 if cat == current_risk else 0.3 for cat in categories]
    
    fig = go.Figure(data=[
        go.Bar(
            x=categories,
            y=values,
            marker=dict(
                color=colors,
                opacity=opacity,
                line=dict(color='#2E5BBA', width=2)
            ),
            text=[f"{risk_levels[cat]['description']}" for cat in categories],
            textposition='outside',
            textfont=dict(size=11, color='#2E5BBA')
        )
    ])
    
    fig.update_layout(
        title=dict(
            text=f"Risk Assessment: {current_risk.upper()}",
            font=dict(size=18, color='#2E5BBA', family="Arial Bold"),
            x=0.5
        ),
        xaxis_title="Risk Level",
        yaxis_title="Urgency (%)",
        height=400,
        margin=dict(l=50, r=50, t=80, b=50),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Arial, sans-serif", size=12, color='#34495E'),
        xaxis=dict(gridcolor='rgba(46, 91, 186, 0.1)'),
        yaxis=dict(gridcolor='rgba(46, 91, 186, 0.1)'),
        showlegend=False
    )
    
    # Add indicator line for current level
    current_value = risk_levels[current_risk]["value"]
    fig.add_hline(
        y=current_value,
        line_dash="dash",
        line_color=risk_levels[current_risk]["color"],
        line_width=3,
        annotation_text=f"Current Level: {current_risk}",
        annotation_position="top right"
    )
    
    return fig

def create_comparison_chart(probabilities: Dict[str, float]) -> go.Figure:
    """
    Creates a comparison chart among the top 3 most probable diseases.
    
    Args:
        probabilities: Dictionary with probabilities per class
        
    Returns:
        Plotly figure
    """
    # Get top 3 diseases
    sorted_diseases = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)[:3]
    
    if len(sorted_diseases) < 2:
        # Empty chart if not enough data
        fig = go.Figure()
        fig.update_layout(title="Insufficient data for comparison")
        return fig
    
    diseases, probs = zip(*sorted_diseases)
    probs_percent = [p * 100 for p in probs]
    colors = [DISEASE_INFO.get(disease, {}).get("color", "#636EFA") for disease in diseases]
    
    # Create pie chart
    fig = go.Figure(data=[
        go.Pie(
            labels=diseases,
            values=probs_percent,
            hole=0.4,
            marker=dict(
                colors=colors,
                line=dict(color='#2E5BBA', width=3)
            ),
            textinfo='label+percent',
            textfont=dict(size=13, color='black', family='Arial Bold'),
            textposition='outside'
        )
    ])
    
    fig.update_layout(
        title=dict(
            text="Top 3 Diagnoses Comparison",
            font=dict(size=18, color='#2E5BBA', family="Arial Bold"),
            x=0.5
        ),
        height=400,
        margin=dict(l=50, r=50, t=80, b=50),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Arial, sans-serif", size=12, color='#34495E'),
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="middle",
            y=0.5,
            xanchor="left",
            x=1.05
        )
    )
    
    # Add annotation in the center
    fig.add_annotation(
        x=0.5, y=0.5,
        text=f"<b>{diseases[0]}</b><br>{probs_percent[0]:.1f}%",
        showarrow=False,
        font=dict(size=16, color='#2E5BBA', family='Arial Bold'),
        align="center"
    )
    
    return fig

def create_severity_timeline() -> go.Figure:
    """
    Creates a timeline chart showing the typical severity evolution.
    
    Returns:
        Plotly figure
    """
    # Simulated temporal evolution data
    timeline_data = {
        "DDays": [0, 7, 14, 21, 30, 60, 90],
        "No Treatment": [30, 45, 60, 75, 85, 90, 95],
        "Early Treatment": [30, 25, 20, 15, 10, 5, 3],
        "Late Treatment": [30, 50, 65, 55, 40, 25, 15]
    }
    
    fig = go.Figure()
    
    # No Treatment line
    fig.add_trace(go.Scatter(
        x=timeline_data["DDays"],
        y=timeline_data["No Treatment"],
        mode='lines+markers',
        name='No Treatment',
        line=dict(color='#E74C3C', width=3, dash='solid'),
        marker=dict(size=8, color='#E74C3C')
    ))
    
    # Early Treatment line
    fig.add_trace(go.Scatter(
        x=timeline_data["DDays"],
        y=timeline_data["Early Treatment"],
        mode='lines+markers',
        name='Early Treatment',
        line=dict(color='#27AE60', width=3, dash='solid'),
        marker=dict(size=8, color='#27AE60')
    ))
    
    # Late Treatment line
    fig.add_trace(go.Scatter(
        x=timeline_data["DDays"],
        y=timeline_data["Late Treatment"],
        mode='lines+markers',
        name='Late Treatment',
        line=dict(color='#F39C12', width=3, dash='dash'),
        marker=dict(size=8, color='#F39C12')
    ))
    
    fig.update_layout(
        title=dict(
            text="Temporal Evolution of Severity",
            font=dict(size=18, color='#2E5BBA', family="Arial Bold"),
            x=0.5
        ),
        xaxis_title="Time (days)",
        yaxis_title="Severity (%)",
        height=400,
        margin=dict(l=50, r=50, t=80, b=50),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Arial, sans-serif", size=12, color='#34495E'),
        xaxis=dict(gridcolor='rgba(46, 91, 186, 0.1)', showgrid=True),
        yaxis=dict(gridcolor='rgba(46, 91, 186, 0.1)', showgrid=True),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        )
    )
    
    return fig

def display_medical_footer():
    """Displays a professional medical footer."""
    st.markdown("""
    <div class="footer">
        <h4 style="margin: 0 0 1rem 0;"> Dermosan - Dermatological Diagnostic System</h4>
        <p style="margin: 0.5rem 0; opacity: 0.9;">
            DERMOSAN Project – UNDC 2025 | Developed with AI ResNet152
        </p>
        <p style="margin: 0.5rem 0; font-size: 0.9rem; opacity: 0.8;">
            🔬 This system is a diagnostic support tool. 
            Always consult with a professional dermatologist.
        </p>
        <div style="margin-top: 1rem; padding-top: 1rem; border-top: 1px solid rgba(255,255,255,0.2);">
            <small style="opacity: 0.7;">
                © 2025 Dermosan | Accuracy: 95% | 10 Detectable Diseases
            </small>
        </div>
    </div>
    """, unsafe_allow_html=True)
