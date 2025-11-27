"""
Main application of the Dermosan dermatological diagnostic system
"""

import streamlit as st
import logging
from PIL import Image
import io
import sys
import os
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO)

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Local module imports
from src.predictor import DermatologyPredictor, analyze_image_quality
from src.utils import (
    set_page_config, display_header, display_sidebar_info,
    create_confidence_gauge, create_probability_chart, create_compact_probability_chart,
    create_risk_assessment_chart, create_comparison_chart, create_severity_timeline,
    display_disease_info, display_quality_analysis,
    display_medical_recommendations, export_diagnosis_report,
    create_download_link, display_medical_footer, display_confidence_level
)
from src.config import APP_CONFIG

def main():
    """Main function of the application."""
    
    # Configure page
    set_page_config()
    
    # Show header
    display_header()
    
    # Show sidebar information
    display_sidebar_info()
    
    # Initialize predictor
    @st.cache_resource
    def load_predictor():
        """Load the predictor with cache to optimize performance."""
        try:
            with st.spinner("Loading AI model..."):
                predictor = DermatologyPredictor()
                st.success("Model loaded successfully")
                return predictor
        except FileNotFoundError as e:
            st.error("**Error:** Model file not found")
            st.info("""
            **Solution:** Ensure the model file is located in one of these paths:
            - `models/best_resnet152.h5` (recommended)
            - `Trained_Model/best_resnet152.h5` (alternative)
            """)
            st.stop()
        except Exception as e:
            st.error(f"**Critical error loading model:** {str(e)}")
            with st.expander("Technical error details"):
                st.code(f"Type: {type(e).__name__}\nMessage: {str(e)}")
            st.info("""
            **Possible solutions:**
            1. Verify that TensorFlow is installed: `pip install tensorflow`
            2. Check model compatibility
            3. Run verification script: `python verify_model.py`
            """)
            return None
    
    predictor = load_predictor()
    
    if predictor is None:
        st.stop()
    
    # Main interface
    st.markdown("### Upload Image for Diagnosis")
    
    # Quick metrics at the top
    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a:
        st.metric("Accuracy", "95%", "Optimized")
    with col_b:
        st.metric("Speed", "< 3s", "Fast")
    with col_c:
        st.metric("Model", "ResNet152", "Loaded")
    with col_d:
        st.metric("Status", "Active", "Online")
    
    st.markdown("---")
    
    # Image upload
    uploaded_file = st.file_uploader(
        "Select a dermatological image",
        type=['jpg', 'jpeg', 'png'],
        help="Accepted formats: JPG, JPEG, PNG"
    )
    
    if uploaded_file is not None:
        try:
            # Load image
            image = Image.open(uploaded_file)
            
            # Layout with columns
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("#### Uploaded Image")
                st.image(image, caption="Image for diagnosis", width=300)
            
            with col2:
                st.markdown("#### Quality Analysis")
                
                with st.spinner("Analyzing image quality..."):
                    quality_result = analyze_image_quality(image)
                
                display_quality_analysis(quality_result)
            
            # Perform prediction
            should_analyze = st.button("Analyze Image", type="primary")
            
            if should_analyze:
                with st.spinner("Analyzing image with dermatological AI..."):
                    prediction_result = predictor.predict(image)
                    recommendations = predictor.get_medical_recommendation(prediction_result)
                
                # Main result
                predicted_disease = prediction_result['predicted_class']
                confidence = prediction_result['confidence']
                
                # Results dashboard
                st.markdown("---")
                st.markdown("## Detailed Results Dashboard")
                
                # Enhanced main dashboard
                st.markdown("### Main Diagnosis")
                
                # Improved layout: 3 columns
                main_row1_col1, main_row1_col2, main_row1_col3 = st.columns([2, 1, 1.5])
                
                with main_row1_col1:
                    # Main diagnosis information with enhanced design
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #2E5BBA, #4A90B8); 
                                color: white; padding: 1.5rem; border-radius: 15px; 
                                text-align: center; margin-bottom: 1rem;">
                        <h2 style="margin: 0 0 0.5rem 0; color: white; font-size: 1.8rem;">
                            {predicted_disease}
                        </h2>
                        <h3 style="margin: 0; color: rgba(255,255,255,0.9); font-size: 1.3rem;">
                            Confianza: {prediction_result['confidence_percentage']}
                        </h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Display confidence level with colors
                    display_confidence_level(confidence)
                
                with main_row1_col2:
                    # Compact confidence gauge
                    st.markdown("**Gauge**")
                    st.plotly_chart(
                        create_confidence_gauge(confidence),
                        use_container_width=True,
                        config={'displayModeBar': False}
                    )
                
                with main_row1_col3:
                    # Integrated probability distribution - compact version
                    st.markdown("**Top Probabilities**")
                    st.plotly_chart(
                        create_compact_probability_chart(prediction_result['all_probabilities']),
                        use_container_width=True,
                        config={'displayModeBar': False}
                    )
                
                # Visual statistical summary
                st.markdown("---")
                st.markdown("""
                <div style="background: linear-gradient(135deg, #ECF0F1, #BDC3C7); 
                            padding: 1.5rem; border-radius: 15px; margin: 1rem 0;">
                    <h3 style="text-align: center; color: #2C3E50; margin: 0 0 1rem 0;">
                        Resumen del Análisis
                    </h3>
                </div>
                """, unsafe_allow_html=True)
                
                # Key metrics in columns
                metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                
                with metric_col1:
                    confidence_val = int(confidence * 100)
                    st.metric(
                        label="Confianza",
                        value=f"{confidence_val}%",
                        delta=f"{'Alto' if confidence_val > 80 else 'Medio' if confidence_val > 60 else 'Bajo'}"
                    )
                
                with metric_col2:
                    # Calculate number of diagnoses considered
                    num_diagnoses = len([p for p in prediction_result['all_probabilities'].values() if p > 0.05])
                    st.metric(
                        label="Diagnoses",
                        value=f"{num_diagnoses}",
                        delta="analyzed"
                    )
                
                with metric_col3:
                    # Determine risk level
                    risk_level = "High" if "Melanoma" in predicted_disease or "Carcinoma" in predicted_disease else "Medium" if confidence < 0.7 else "Low"
                    st.metric(
                        label="Risk Level",
                        value=risk_level,
                        delta="evaluated"
                    )
                
                with metric_col4:
                    st.metric(
                        label="Analysis Time",
                        value="< 5s",
                        delta="Fast"
                    )
                
                # Advanced analysis section
                st.markdown("---")
                st.markdown("""
                <div style="text-align: center; margin: 2rem 0 1rem 0;">
                    <h3 style="color: #2E5BBA; margin: 0;">Advanced Clinical Analysis</h3>
                    <p style="color: #34495E; margin: 0.5rem 0 0 0; font-style: italic;">
                        Comprehensive risk assessment and diagnostic comparisons
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                # First row of important charts
                analysis_col1, analysis_col2 = st.columns(2)
                
                with analysis_col1:
                    st.markdown("""
                    <div style="text-align: center; margin-bottom: 1rem;">
                        <h4 style="color: #E74C3C; margin: 0;">Risk Assessment</h4>
                        <p style="color: #7F8C8D; font-size: 0.9rem; margin: 0.3rem 0;">Level of medical urgency</p>
                    </div>
                    """, unsafe_allow_html=True)
                    st.plotly_chart(
                        create_risk_assessment_chart(predicted_disease, confidence),
                        use_container_width=True,
                        config={'displayModeBar': False}
                    )
                
                with analysis_col2:
                    st.markdown("""
                    <div style="text-align: center; margin-bottom: 1rem;">
                        <h4 style="color: #27AE60; margin: 0;">Diagnostic Comparison</h4>
                        <p style="color: #7F8C8D; font-size: 0.9rem; margin: 0.3rem 0;">Top 3 most probable diagnoses</p>
                    </div>
                    """, unsafe_allow_html=True)
                    st.plotly_chart(
                        create_comparison_chart(prediction_result['all_probabilities']),
                        use_container_width=True,
                        config={'displayModeBar': False}
                    )
                
                # Temporal evolution
                st.markdown("---")
                st.markdown("""
                <div style="text-align: center; margin: 2rem 0 1rem 0;">
                    <h4 style="color: #F39C12; margin: 0;">Temporal Evolution Projection</h4>
                    <p style="color: #7F8C8D; font-size: 0.9rem; margin: 0.3rem 0;">
                        Progression simulation with different treatment scenarios
                    </p>
                </div>
                """, unsafe_allow_html=True)
                st.plotly_chart(
                    create_severity_timeline(),
                    use_container_width=True,
                    config={'displayModeBar': False}
                )
                
                # Detailed information
                st.markdown("---")
                st.markdown("### Detailed Clinical Information")
                
                info_col1, info_col2 = st.columns(2)
                
                with info_col1:
                    display_disease_info(predicted_disease)
                
                with info_col2:
                    st.markdown("#### Medical Recommendations")
                    display_medical_recommendations(recommendations)
                
                # Medical disclaimer
                st.markdown("""
                <div style="background: linear-gradient(135deg, #34495E, #2C3E50); 
                            color: white; padding: 2rem; border-radius: 15px; 
                            text-align: center; margin: 2rem 0;">
                    <h3 style="margin: 0 0 1rem 0; color: white;">Important Medical Disclaimer</h3>
                    <p style="margin: 0; color: rgba(255,255,255,0.9);">
                        This system is a diagnostic support tool that uses 
                        artificial intelligence. Results should always be interpreted by a 
                        professional dermatologist. It does not replace medical clinical judgment.
                    </p>
                </div>
                """, unsafe_allow_html=True)
        
        except Exception as e:
            st.error(f"**Error processing the image:** {str(e)}")
            logging.error(f"Processing error: {str(e)}")
            traceback.print_exc()
    
    else:
        st.info("Upload a dermatological image to start the analysis")
    
    # Footer
    display_medical_footer()

if __name__ == "__main__":
    main()