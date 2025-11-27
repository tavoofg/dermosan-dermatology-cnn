"""
Specific script for testing the loading of the Dermosan model
"""

import os
import sys
import tensorflow as tf
import numpy as np
from PIL import Image

# Add directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def verify_model_file():
    """Verify that the model file exists and its properties."""
    model_path = "Trained_Model/best_resnet152.h5"
    
    print("🔍 Verifying model file...")
    print(f"Path: {model_path}")
    
    if os.path.exists(model_path):
        size_mb = os.path.getsize(model_path) / (1024 * 1024)
        print(f"✅ File found - Size: {size_mb:.1f} MB")
        return True
    else:
        print(f"❌ File not found at: {os.path.abspath(model_path)}")
        return False

def test_loading_methods():
    """Test different methods of loading the model."""
    model_path = "Trained_Model/best_resnet152.h5"
    
    print("\n🧪 Testing model loading methods...")
    
    # MMethod 1: Normal load
    print("\n[MMethod 1] Normal load...")
    try:
        model = tf.keras.models.load_model(model_path)
        print("✅ MMethod 1 successful")
        print(f"   - Input: {model.input_shape}")
        print(f"   - Output: {model.output_shape}")
        print(f"   - Parameters: {model.count_params():,}")
        return model
    except Exception as e:
        print(f"❌ MMethod 1 failed: {str(e)}")
    
    # MMethod 2: Load without compiling
    print("\n[MMethod 2] Load without compiling...")
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        print("✅ MMethod 2 successful")
        
        # Recompile
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        print("✅ Model recompiled")
        print(f"   - Input: {model.input_shape}")
        print(f"   - Output: {model.output_shape}")
        print(f"   - Parameters: {model.count_params():,}")
        return model
    except Exception as e:
        print(f"❌ MMethod 2 failed: {str(e)}")
    
    # MMethod 3: Rebuild architecture
    print("\n[MMethod 3] Rebuild architecture...")
    try:
        model = rebuild_model_and_load_weights(model_path)
        print("✅ MMethod 3 successful")
        return model
    except Exception as e:
        print(f"❌ MMethod 3 failed: {str(e)}")
    
    return None

def rebuild_model_and_load_weights(model_path):
    """Rebuild the model architecture and load weights."""
    from tensorflow.keras.applications import ResNet152
    from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
    from tensorflow.keras.models import Model
    
    print("   - Creating ResNet152 base...")
    base = ResNet152(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )
    
    print("   - Configuring trainable layers...")
    for layer in base.layers[:-50]:
        layer.trainable = False
    
    print("   - Adding classification layers...")
    x = GlobalAveragePooling2D()(base.output)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.35)(x)
    x = Dense(256, activation='relu')(x)
    out = Dense(10, activation='softmax')(x)  # 10 classes
    
    model = Model(inputs=base.input, outputs=out)
    
    print("   - Compiling model...")
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("   - Loading weights...")
    model.load_weights(model_path)
    
    return model

def test_prediction(model):
    """Test a simple prediction with the model."""
    print("\n🎯 Testing prediction...")
    
    try:
        # Create test image
        test_image = np.random.rand(1, 224, 224, 3).astype(np.float32)
        test_image = (test_image - 0.5) * 2  # Normalize as ResNet
        
        print("   - Making prediction...")
        predictions = model.predict(test_image, verbose=0)
        
        print(f"✅ Prediction successful")
        print(f"   - Output shape: {predictions.shape}")
        print(f"   - Sum of probabilities: {predictions.sum():.3f}")
        print(f"   - Predicted class: {np.argmax(predictions[0])}")
        print(f"   - Maximum confidence: {np.max(predictions[0]):.3f}")
        
        return True
    except Exception as e:
        print(f"❌ Prediction error: {str(e)}")
        return False

def main():
    print("🏥 DERMOSAN - MODEL VERIFIER")
    print("=" * 50)
    
    # Verify file
    if not verify_model_file():
        print("\n❌ Cannot continue without the model file")
        return
    
    # Test loading
    model = test_loading_methods()
    
    if model is None:
        print("\n❌ Could not load the model with any method")
        return
    
    # Test prediction
    if test_prediction(model):
        print("\n🎉 MODEL VERIFIED SUCCESSFULLY!")
        print("\nThe model is ready to use in the Streamlit application")
    else:
        print("\n⚠️ Model loaded but with prediction issues")
    
    print("\n" + "=" * 50)
    print("To run the application: streamlit run app.py")

if __name__ == "__main__":
    main()
