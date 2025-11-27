"""
Custom CSS styles for Dermosan
"""

def get_custom_css():
    """Returns custom CSS for the application."""
    return """
    <style>
    /* CSS variables for medical colors */
    :root {
        --primary-blue: #2E5BBA;
        --secondary-blue: #4A90B8;
        --medical-green: #27AE60;
        --medical-gray: #34495E;
        --medical-red: #E74C3C;
        --medical-orange: #F39C12;
        --medical-info: #3498DB;
        --light-gray: #ECF0F1;
        --white: #FFFFFF;
        --shadow-color: rgba(46, 91, 186, 0.1);
    }

    /* General styles */
    .stApp {
        background-color: var(--light-gray);
    }

    /* Sidebar customization */
    .css-1d391kg {
        background: linear-gradient(180deg, var(--white) 0%, #F8F9FA 100%);
        border-right: 2px solid var(--primary-blue);
    }

    /* Custom buttons */
    .stButton > button {
        background: linear-gradient(135deg, var(--primary-blue), var(--secondary-blue));
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.5rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px var(--shadow-color);
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px var(--shadow-color);
    }

    /* Custom metrics */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 12px var(--shadow-color);
        border-left: 4px solid var(--primary-blue);
        margin: 1rem 0;
    }

    /* Custom file uploader */
    .css-1cpxqw2 {
        border: 2px dashed var(--primary-blue);
        border-radius: 10px;
        background-color: rgba(46, 91, 186, 0.05);
    }

    /* Custom alerts */
    .stAlert {
        border-radius: 10px;
        border-left: 4px solid var(--medical-green);
    }

    /* Result containers */
    .result-container {
        background: white;
        border-radius: 15px;
        padding: 1.5rem;
        box-shadow: 0 4px 20px var(--shadow-color);
        margin: 1rem 0;
        border: 1px solid rgba(46, 91, 186, 0.1);
    }

    /* Confidence text */
    .confidence-high {
        color: var(--medical-green);
        font-weight: 600;
    }

    .confidence-medium {
        color: var(--medical-orange);
        font-weight: 600;
    }

    .confidence-low {
        color: var(--medical-red);
        font-weight: 600;
    }

    /* Custom tables */
    .stDataFrame {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 2px 8px var(--shadow-color);
    }

    /* Custom sidebar */
    .sidebar-content {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 8px var(--shadow-color);
    }

    /* Status indicators */
    .status-indicator {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        margin-right: 8px;
    }

    .status-active {
        background-color: var(--medical-green);
        animation: pulse 2s infinite;
    }

    .status-warning {
        background-color: var(--medical-orange);
    }

    .status-error {
        background-color: var(--medical-red);
    }

    /* Pulse animation */
    @keyframes pulse {
        0% {
            box-shadow: 0 0 0 0 rgba(39, 174, 96, 0.7);
        }
        70% {
            box-shadow: 0 0 0 10px rgba(39, 174, 96, 0);
        }
        100% {
            box-shadow: 0 0 0 0 rgba(39, 174, 96, 0);
        }
    }

    /* Custom plots */
    .plotly-graph-div {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 2px 12px var(--shadow-color);
    }

    /* Custom footer */
    .footer {
        background: linear-gradient(135deg, var(--medical-gray), var(--primary-blue));
        color: white;
        text-align: center;
        padding: 2rem;
        border-radius: 10px;
        margin-top: 2rem;
    }

    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }

    ::-webkit-scrollbar-track {
        background: var(--light-gray);
    }

    ::-webkit-scrollbar-thumb {
        background: var(--primary-blue);
        border-radius: 4px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: var(--secondary-blue);
    }

    /* Responsive design */
    @media (max-width: 768px) {
        .metric-card {
            margin: 0.5rem 0;
            padding: 1rem;
        }
        
        .result-container {
            padding: 1rem;
            margin: 0.5rem 0;
        }
    }
    </style>
    """

def apply_custom_styling():
    """Applies custom styles to the application."""
    import streamlit as st
    st.markdown(get_custom_css(), unsafe_allow_html=True)

def create_metric_card(title: str, value: str, description: str = "", color: str = "primary"):
    """
    Creates a custom metric card.
    
    Args:
        title: Metric title
        value: Main value
        description: Additional description
        color: Theme color (primary, success, warning, danger)
    """
    color_map = {
        "primary": "#2E5BBA",
        "success": "#27AE60", 
        "warning": "#F39C12",
        "danger": "#E74C3C"
    }
    
    border_color = color_map.get(color, color_map["primary"])
    
    return f"""
    <div class="metric-card" style="border-left-color: {border_color};">
        <h3 style="color: {border_color}; margin: 0 0 0.5rem 0; font-size: 1.1rem;">{title}</h3>
        <h2 style="color: #34495E; margin: 0; font-size: 2rem; font-weight: 700;">{value}</h2>
        {f'<p style="color: #7F8C8D; margin: 0.5rem 0 0 0; font-size: 0.9rem;">{description}</p>' if description else ''}
    </div>
    """

def create_status_indicator(status: str, text: str):
    """
    Creates a status indicator with animation.
    
    Args:
        status: Status type (active, warning, error)
        text: Text to display
    """
    return f"""
    <div style="display: flex; align-items: center; margin: 0.5rem 0;">
        <span class="status-indicator status-{status}"></span>
        <span style="color: #34495E; font-weight: 500;">{text}</span>
    </div>
    """
