# 📁 DERMOSAN Project Structure

## 🏗️ Main Architecture

```
dermosan/
├── 📄 app.py                         # Main Streamlit application
├── 📄 README.md                      # Project documentation
├── 📄 test_system.py                 # System testing script
├── 📄 verify_model.py            # Model verification script
├── 📄 .gitignore                     # Files ignored by Git
│
├── 🗂️ src/                           # Core modules
│   ├── 📄 __init__.py                # Package initialization
│   ├── 📄 config.py                  # System configurations
│   ├── 📄 predictor.py               # AI prediction logic
│   └── 📄 utils.py                   # Utilities and UI components
│
├── 🗂️ Trained Model/                 # Trained model (.h5) goes here after training
│   └── 📄 best_resnet152.h5          # Final ResNet152 model (generated after training)
│
└── 🗂️ Training Code/                 # Training scripts
    ├── 📄 train_resnet152.py         # AI model training script
    └── 🗂️ IMG_CLASSES/               # Kaggle dataset goes here: https://www.kaggle.com/datasets/ismailpromus/skin-diseases-image-dataset 

```


## Main Files

### 📄 **app.py**

* **Purpose:** Main interface of the web application
* **Technology:** Streamlit + Plotly
* **Features:** Medical dashboard, image analysis, report generation

### 📄 **src/predictor.py**

* **Purpose:** Dermatological prediction engine
* **Technology:** TensorFlow + ResNet152
* **Features:** Quality analysis, AI prediction, medical recommendations

### 📄 **src/utils.py**

* **Purpose:** Interface components and utilities
* **Technology:** Streamlit + Plotly
* **Features:** Medical charts, validations, result export

### 📄 **src/config.py**

* **Purpose:** Centralized configurations
* **Features:** Model parameters, confidence thresholds, global constants

---

## Deployment Files

### ✅ **Required for Production:**

* `app.py` – Main application
* `src/` – System modules
* `Trained_Model/best_resnet152.h5` – Trained AI model
* `requirements.txt` – Environment dependencies

### 🧩 **Development Files:**

* `Training_Code/` – Scripts for training or retraining
* `IMG_CLASSES/` – Kaggle dataset for training
* `test_system.py` – System functional tests
* `verify_model.py` – Trained model validation

### 🚫 **Excluded from the Repository:**

* `dermosan_venv/` – Local virtual environment
* `__pycache__/` – Python cache directories
* `*.log` – Log files
* Temporary files and automatic backups

---

## Applied Optimizations

1. **✅ Reorganized structure:** Training files moved to `Training_Code/`
2. **✅ Removed obsolete and duplicate files**
3. **✅ Cache cleanup:** Deleted `__pycache__` directories
4. **✅ Updated `.gitignore`:** Added project-specific exclusions
5. **✅ Standardized folder and file naming conventions**

---

## 📊 Project Metrics

* **Main files:** 4 Python files
* **Modules in `src/`:** 4 files
* **Model size:** ~500 MB (`best_resnet152.h5`)
* **Key dependencies:** TensorFlow, Streamlit, Plotly, Pillow
* **Estimated model accuracy:** ~95%

---

*Optimized structure – October 2025*
