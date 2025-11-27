"""
Test script to verify that the Dermosan system is functioning correctly
"""

import os
import sys
import traceback
from PIL import Image
import numpy as np

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_model_loading():
    """Test model loading."""
    print("🔍 Testing model loading...")
    try:
        from src.predictor import DermatologyPredictor
        predictor = DermatologyPredictor()
        print("✅ Model loaded successfully")
        return True
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        traceback.print_exc()
        return False

def test_image_processing():
    """Test image processing."""
    print("\n🖼️ Testing image processing...")
    try:
        from src.predictor import analyze_image_quality
        
        # Create test image
        test_image = Image.new('RGB', (224, 224), color='red')
        
        # Analyze quality
        quality_result = analyze_image_quality(test_image)
        print(f"✅ Quality analysis: {quality_result['quality_score']}/100")
        return True
    except Exception as e:
        print(f"❌ Error in image processing: {str(e)}")
        traceback.print_exc()
        return False

def test_prediction():
    """Test a complete prediction."""
    print("\n🎯 Testing complete prediction...")
    try:
        from src.predictor import DermatologyPredictor
        
        predictor = DermatologyPredictor()
        
        # Create test image
        test_image = Image.new('RGB', (224, 224), color='blue')
        
        # Make prediction
        result = predictor.predict(test_image)
        
        print(f"✅ Successful prediction:")
        print(f"   - Class: {result['predicted_class']}")
        print(f"   - Confidence: {result['confidence_percentage']}")
        print(f"   - Level: {result['confidence_level']}")
        
        # Test recommendations
        recommendations = predictor.get_medical_recommendation(result)
        print(f"   - Urgency: {recommendations['urgency']}")
        
        return True
    except Exception as e:
        print(f"❌ Error in prediction: {str(e)}")
        traceback.print_exc()
        return False

def test_streamlit_components():
    """Test Streamlit components."""
    print("\n🎨 Testing interface components...")
    try:
        from src.utils import create_confidence_gauge, create_probability_chart
        from src.config import DISEASE_CLASSES
        
        # Test gauge
        fig_gauge = create_confidence_gauge(0.85)
        print("✅ Confidence gauge created")
        
        # Test probability chart
        test_probs = {disease: np.random.random() for disease in DISEASE_CLASSES.values()}
        fig_chart = create_probability_chart(test_probs)
        print("✅ Probability chart created")
        
        return True
    except Exception as e:
        print(f"❌ Error in components: {str(e)}")
        traceback.print_exc()
        return False

def check_dependencies():
    """Check main dependencies."""
    print("\n📦 Checking dependencies...")
    
    dependencies = [
        'tensorflow', 'streamlit', 'numpy', 'pandas', 
        'plotly', 'PIL', 'cv2', 'sklearn'
    ]
    
    missing = []
    
    for dep in dependencies:
        try:
            if dep == 'PIL':
                import PIL
            elif dep == 'cv2':
                import cv2
            elif dep == 'sklearn':
                import sklearn
            else:
                __import__(dep)
            print(f"✅ {dep}")
        except ImportError:
            print(f"❌ {dep} - NOT INSTALLED")
            missing.append(dep)
    
    return len(missing) == 0

def check_file_structure():
    """Check file structure."""
    print("\n📁 Checking file structure...")
    
    required_files = [
        "Trained_Model/best_resnet152.h5",
        "src/config.py",
        "src/predictor.py", 
        "src/utils.py",
        "app.py",
        "requirements.txt"
    ]
    
    missing = []
    
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - NOT FOUND")
            missing.append(file_path)
    
    return len(missing) == 0

def main():
    """Main testing function."""
    print("🏥 DERMOSAN - TESTING SYSTEM")
    print("=" * 50)
    
    tests = [
        ("File Structure", check_file_structure),
        ("Dependencies", check_dependencies),
        ("Model Loading", test_model_loading),
        ("Image Processing", test_image_processing),
        ("Full Prediction", test_prediction),
        ("Interface Components", test_streamlit_components)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'=' * 20} {test_name.upper()} {'=' * 20}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Unexpected error in {test_name}: {str(e)}")
            results.append((test_name, False))
    
    # Final summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:.<30} {status}")
        if result:
            passed += 1
    
    print(f"\nResult: {passed}/{len(results)} successful tests")
    
    if passed == len(results):
        print("\n🎉 ALL TESTS PASSED! The system is ready to use.")
        print("\nTo run the application:")
        print("   streamlit run app.py")
    else:
        print(f"\n⚠️ {len(results) - passed} tests failed. Check errors above.")
        
        if not results[0][1]:  # File Structure
            print("\n💡 Suggestion: Verify that all files are in place")
        if not results[1][1]:  # Dependencies
            print("\n💡 Suggestion: Install dependencies with: pip install -r requirements.txt")

if __name__ == "__main__":
    main()
