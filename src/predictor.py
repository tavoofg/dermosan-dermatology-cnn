"""
Prediction module for the dermatological diagnostic system
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet import preprocess_input
from PIL import Image
import cv2
from typing import Dict, Tuple, List
import logging

from src.config import MODEL_PATH, MODEL_PATH_FALLBACK, IMG_SIZE, DISEASE_CLASSES, CONFIDENCE_THRESHOLDS

class DermatologyPredictor:
    """
    Class for making predictions of dermatological diseases
    using the trained ResNet152 model.
    """
    
    def __init__(self):
        """Initialize the predictor by loading the model."""
        self.model = None
        self.load_model()
        
    def load_model(self):
        """Load the trained model."""
        try:
            # Try first path (corrected)
            if os.path.exists(MODEL_PATH):
                model_path = MODEL_PATH
            # Try fallback path
            elif os.path.exists(MODEL_PATH_FALLBACK):
                model_path = MODEL_PATH_FALLBACK
                logging.warning(f"Using fallback path: {MODEL_PATH_FALLBACK}")
            else:
                raise FileNotFoundError(f"Model not found in {MODEL_PATH} or {MODEL_PATH_FALLBACK}")
            
            # Try loading with compatible configurations
            try:
                # MMethod 1: Normal load
                self.model = tf.keras.models.load_model(model_path)
                logging.info(f"Model successfully loaded from {model_path}")
            except Exception as e1:
                logging.warning(f"Method 1 failed: {str(e1)}")
                try:
                    # MMethod 2: Load with compile=False
                    self.model = tf.keras.models.load_model(model_path, compile=False)
                    # Recompile the model manually
                    self.model.compile(
                        optimizer='adam',
                        loss='sparse_categorical_crossentropy',
                        metrics=['accuracy']
                    )
                    logging.info(f"Model loaded with compile=False from {model_path}")
                except Exception as e2:
                    logging.warning(f"Method 2 failed: {str(e2)}")
                    # MMethod 3: Load weights only and rebuild architecture
                    self._load_model_weights_only(model_path)
        except Exception as e:
            logging.error(f"Error loading model: {str(e)}")
            raise
    
    def _load_model_weights_only(self, model_path):
        """Alternative method: rebuild model and load weights only."""
        try:
            from tensorflow.keras.applications import ResNet152
            from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
            from tensorflow.keras.models import Model
            
            logging.info("Attempting to rebuild model architecture...")
            
            # Rebuild model architecture
            base = ResNet152(
                weights='imagenet',
                include_top=False,
                input_shape=(*IMG_SIZE, 3)
            )
            
            # Make the last 50 layers trainable (as in the original script)
            for layer in base.layers[:-50]:
                layer.trainable = False
            
            # Add classification layers
            x = GlobalAveragePooling2D()(base.output)
            x = Dense(512, activation='relu')(x)
            x = Dropout(0.35)(x)
            x = Dense(256, activation='relu')(x)
            out = Dense(len(DISEASE_CLASSES), activation='softmax')(x)
            
            self.model = Model(inputs=base.input, outputs=out)
            
            # Compile
            self.model.compile(
                optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
            # Load weights only
            self.model.load_weights(model_path)
            logging.info("Model rebuilt and weights loaded successfully")
            
        except Exception as e:
            logging.error(f"Error rebuilding model: {str(e)}")
            raise
    
    def preprocess_image(self, image: Image.Image) -> np.ndarray:
        """
        Preprocess the image for the model.
        
        Args:
            image: PIL Image
            
        Returns:
            Preprocessed numpy array
        """
        try:
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize
            image = image.resize(IMG_SIZE)
            
            # Convert to numpy array
            img_array = np.array(image)
            
            # Apply ResNet preprocessing
            img_array = preprocess_input(img_array)
            
            # Add batch dimension
            img_array = np.expand_dims(img_array, axis=0)
            
            return img_array
            
        except Exception as e:
            logging.error(f"Error en preprocesamiento: {str(e)}")
            raise
    
    def predict(self, image: Image.Image) -> Dict:
        """
        Make a prediction on an image.
        
        Args:
            image: PIL Image
            
        Returns:
            Dictionary with prediction results
        """
        try:
            if self.model is None:
                raise ValueError("Model not loaded")
            
            # Preprocess image
            processed_image = self.preprocess_image(image)
            
            # Make prediction
            predictions = self.model.predict(processed_image, verbose=0)
            probabilities = predictions[0]
            
            # Get predicted class
            predicted_class_idx = np.argmax(probabilities)
            predicted_class = DISEASE_CLASSES[predicted_class_idx]
            confidence = float(probabilities[predicted_class_idx])
            
            # Determine confidence level
            confidence_level = self._get_confidence_level(confidence)
            
            # Get top 3 predictions
            top_3_indices = np.argsort(probabilities)[-3:][::-1]
            top_3_predictions = [
                {
                    "disease": DISEASE_CLASSES[idx],
                    "probability": float(probabilities[idx]),
                    "percentage": f"{probabilities[idx]*100:.1f}%"
                }
                for idx in top_3_indices
            ]
            
            return {
                "predicted_class": predicted_class,
                "confidence": confidence,
                "confidence_level": confidence_level,
                "confidence_percentage": f"{confidence*100:.1f}%",
                "top_3_predictions": top_3_predictions,
                "all_probabilities": {
                    DISEASE_CLASSES[i]: float(prob) 
                    for i, prob in enumerate(probabilities)
                }
            }
            
        except Exception as e:
            logging.error(f"Error in prediction: {str(e)}")
            raise
    
    def _get_confidence_level(self, confidence: float) -> str:
        """
        Determines the confidence level based on the threshold.
        
        Args:
            confidence: Confidence value
            
        Returns:
            Confidence level as a string
        """
        if confidence >= CONFIDENCE_THRESHOLDS["high"]:
            return "Alta"
        elif confidence >= CONFIDENCE_THRESHOLDS["medium"]:
            return "Media"
        elif confidence >= CONFIDENCE_THRESHOLDS["low"]:
            return "Baja"
        else:
            return "Muy Baja"
    
    def get_medical_recommendation(self, prediction_result: Dict) -> Dict:
        """
        Generates medical recommendations based on the prediction.
        
        Args:
            prediction_result: Prediction result
            
        Returns:
            Dictionary with recommendations
        """
        predicted_class = prediction_result["predicted_class"]
        confidence_level = prediction_result["confidence_level"]
        
        # Recommendations based on confidence
        if confidence_level == "High":
            urgency = "Recommended consultation"
            action = "Schedule an appointment with a dermatologist for confirmation"
        elif confidence_level == "Medium":
            urgency = "Additional evaluation required"
            action = "A second opinion and possible biopsy are recommended"
        else:
            urgency = "Uncertain diagnosis"
        action = "Requires immediate in-person clinical evaluation"

        
        # Specific recommendations for severe conditions
        if "Melanoma" in predicted_class and confidence_level in ["High", "Medium"]:
            urgency = "URGENT - Immediate attention"
            action = "Refer to dermatologic oncologist immediately"
        elif "Carcinoma" in predicted_class and confidence_level in ["High", "Medium"]:
            urgency = "Priority"
            action = "Schedule biopsy and oncologic evaluation"
        
        return {
            "urgency": urgency,
            "recommended_action": action,
            "follow_up": "Follow-up in 2-4 weeks depending on evolution"
        }

def analyze_image_quality(image: Image.Image) -> Dict:
    """
    Analyzes the quality of the image for diagnosis.
    
    Args:
        image: PIL Image
        
    Returns:
        Dictionary with quality metrics
    """
    try:
        # Convert to numpy array
        img_array = np.array(image)
        
        # 1. CRITICAL VALIDATION: Verify that it is a valid medical image
        validation_result = validate_medical_image(img_array)
        if not validation_result["is_valid"]:
            return {
                "quality_score": 0,
                "blur_score": 0,
                "brightness": 0,
                "contrast": 0,
                "resolution": f"{image.size[0]}x{image.size[1]}",
                "issues": validation_result["issues"],
                "is_suitable": False,
                "is_medical_image": False,
                "validation_error": validation_result["error_type"]
            }
        
        # 2. Calculate basic quality metrics
        blur_score = cv2.Laplacian(cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var()
        brightness = np.mean(img_array)
        contrast = np.std(img_array)
        
        # 3. Quality assessment
        quality_score = 0
        issues = []
        
        # Assess sharpness
        if blur_score < 100:
            issues.append("Blurry image - consider retaking photo")
        else:
            quality_score += 25
            
        # Assess brightness
        if brightness < 50:
            issues.append("Image too dark")
        elif brightness > 200:
            issues.append("Image too bright")
        else:
            quality_score += 25
            
        # Assess contrast
        if contrast < 30:
            issues.append("Low contrast")
        else:
            quality_score += 25
            
        # Assess resolution
        width, height = image.size
        if width < 224 or height < 224:
            issues.append("Resolution too low")
        else:
            quality_score += 25
            
        return {
            "quality_score": quality_score,
            "blur_score": blur_score,
            "brightness": brightness,
            "contrast": contrast,
            "resolution": f"{width}x{height}",
            "issues": issues,
            "is_suitable": quality_score >= 75,
            "is_medical_image": True,
            "validation_error": None
        }
        
    except Exception as e:
        logging.error(f"Error in quality analysis: {str(e)}")
        return {
            "quality_score": 0,
            "issues": ["Error analyzing the image"],
            "is_suitable": False,
            "is_medical_image": False,
            "validation_error": "processing_error"
        }

def validate_medical_image(img_array: np.ndarray) -> Dict:
    """
    Validates that the image is appropriate for dermatological analysis.
    
    Args:
        img_array: Numpy array of the image
        
    Returns:
        Dictionary with validation result
    """
    try:
        # 1. Verify that it is not a text/document
        if is_text_document(img_array):
            return {
                "is_valid": False,
                "error_type": "text_document",
                "issues": [
                    "🚫 INVALID IMAGE: A text document was detected",
                    "📋 This system only analyzes skin/dermatological lesion photographs",
                    "📸 Please upload a real dermatological image"
                ]
            }
        
        # 2. Verify that it has medical image characteristics
        if not has_skin_characteristics(img_array):
            return {
                "is_valid": False,
                "error_type": "not_medical",
                "issues": [
                    "🚫 NOT A MEDICAL IMAGE: No skin characteristics detected",
                    "🏥 This system is designed for dermatological images",
                    "📸 Please upload a clear photograph of skin or a skin lesion"
                ]
            }
        
        # 3. Verify that it is not a screenshot
        if is_screenshot(img_array):
            return {
                "is_valid": False,
                "error_type": "screenshot",
                "issues": [
                    "🚫 SCREENSHOT DETECTED",
                    "📱 Screenshots or document images are not allowed",
                    "📸 Use a camera to photograph the skin directly"
                ]
            }
        
        return {
            "is_valid": True,
            "error_type": None,
            "issues": []
        }
        
    except Exception as e:
        return {
            "is_valid": False,
            "error_type": "validation_error",
            "issues": [f"Validation error: {str(e)}"]
        }

def is_text_document(img_array: np.ndarray) -> bool:
    """Detects if the image is a text document."""
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # 1. Detect a lot of text (well-defined white/black areas)
        binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)[1]
        white_ratio = np.sum(binary == 255) / binary.size
        
        # 2. Detect text patterns (long horizontal lines)
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))  # More restrictive
        horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)
        horizontal_ratio = np.sum(horizontal_lines > 0) / horizontal_lines.size
        
        # 3. Detect real text using character detection
        # Look for small rectangular regions that could be letters
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        text_like_regions = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if 50 < area < 500:  # Typical character size
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h if h > 0 else 0
                if 0.1 < aspect_ratio < 3:  # Typical letter aspect ratio
                    text_like_regions += 1
        
        text_density = text_like_regions / (gray.shape[0] * gray.shape[1] / 10000)  # Normalize by area
        
        # 4. Detect very straight and organized edges (typical of documents)
        edges = cv2.Canny(gray, 50, 150)
        edge_ratio = np.sum(edges > 0) / edges.size
        
        # MORE STRICT CRITERIA for text document
        # Only consider document if MULTIPLE indicators are present
        strong_indicators = 0
        
        if white_ratio > 0.85:  # Very white background (more restrictive)
            strong_indicators += 1
        if horizontal_ratio > 0.15:  # Many long horizontal lines
            strong_indicators += 1  
        if text_density > 5:  # High density of text-like regions
            strong_indicators += 1
        if edge_ratio > 0.2 and white_ratio > 0.7:  # Many edges + white background
            strong_indicators += 1
            
        # Needs at least 2 strong indicators to be considered a document
        is_document = strong_indicators >= 2
        
        return is_document
        
    except:
        return False

def has_skin_characteristics(img_array: np.ndarray) -> bool:
    """Checks if the image has typical skin characteristics."""
    try:
        # Convert to HSV for color analysis
        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        
        # 1. Typical human skin color ranges (broader)
        skin_ranges = [
            # Very light skin
            ([0, 10, 60], [25, 255, 255]),
            # Light to medium skin
            ([0, 15, 30], [30, 255, 255]),
            # Medium to dark skin
            ([5, 25, 20], [25, 255, 200]),
            # Inflamed/reddened skin
            ([0, 30, 80], [15, 255, 255])
        ]
        
        skin_pixels = 0
        total_pixels = img_array.shape[0] * img_array.shape[1]
        
        for lower, upper in skin_ranges:
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            skin_pixels += np.sum(mask > 0)
        
        skin_ratio = skin_pixels / total_pixels
        
        # 2. Check organic vs geometric texture
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Detect very straight and long lines (undesired in skin)
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=150)  # Higher threshold
        straight_lines = 0
        
        if lines is not None:
            for line in lines:
                rho, theta = line[0]
                # Only count very horizontal or vertical lines (documents/UI)
                if abs(theta) < 0.1 or abs(theta - np.pi/2) < 0.1 or abs(theta - np.pi) < 0.1:
                    straight_lines += 1
        
        has_many_straight_lines = straight_lines > 15  # More tolerant
        
        # 3. Check if the image is too uniform (typical of screenshots)
        std_dev = np.std(gray)
        is_too_uniform = std_dev < 15
        
        # The image has skin characteristics if:
        # - It has skin tones OR
        # - It does not have many perfect straight lines AND is not very uniform
        has_skin_tones = skin_ratio > 0.05  # More tolerant
        not_geometric = not has_many_straight_lines and not is_too_uniform
        
        return has_skin_tones or not_geometric
        
    except:
        return True  # In case of error, allow analysis

def is_screenshot(img_array: np.ndarray) -> bool:
    """Detects if it is a user interface screenshot."""
    try:
        # 1. Detect very defined edges (typical of UI)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        
        # 2. Detect perfect rectangles
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        perfect_rects = 0
        
        for contour in contours:
            if cv2.contourArea(contour) > 1000:  # Only large contours
                approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
                if len(approx) == 4:  # It's a rectangle
                    perfect_rects += 1
        
        # 3. Check typical UI colors (a lot of white/gray)
        unique_colors = len(np.unique(img_array.reshape(-1, img_array.shape[-1]), axis=0))
        
        # It's a screenshot if it has many perfect rectangles and few unique colors
        return perfect_rects > 5 and unique_colors < 50
        
    except:
        return False
