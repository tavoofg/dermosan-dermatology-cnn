# Dermosan - Dermatological Diagnostic System

DERMOSAN: Development of an interactive software tool in Streamlit for dermatological diagnosis assisted by convolutional neural networks (CNNs)

## 🧠 Project Description

Official repository of the DERMOSAN project, a dermatological diagnostic tool powered by artificial intelligence. It integrates a ResNet152 model trained on 27,153 dermatological images and a Streamlit interface for real-time image upload, analysis, and results visualization.

### 🩹 Diagnosed Diseases

1. **Eczema** *(1,677 images)*
2. **Melanoma** *(3,140 images)*
3. **Atopic Dermatitis** *(1,257 images)*
4. **Basal Cell Carcinoma (BCC)** *(3,323 images)*
5. **Melanocytic Nevi (NV)** *(7,970 images)*
6. **Benign Keratosis-like Lesions (BKL)** *(2,624 images)*
7. **Psoriasis / Lichen Planus & related diseases** *(2,055 images)*
8. **Seborrheic Keratoses & other benign tumors** *(1,847 images)*
9. **Tinea / Candidiasis & other fungal infections** *(1,702 images)*
10. **Warts / Molluscum & other viral infections** *(2,103 images)*

**Overall total:** *27,153 images*


## 🚀 Main Features

* **Deep Learning Model**: ResNet152 with transfer learning
* **Intuitive Interface**: Developed with Streamlit
* **Image Quality Analysis**: Automatic suitability evaluation
* **Differential Diagnosis**: Top 3 predictions with probabilities
* **Clinical Recommendations**: Medical suggestions based on confidence
* **Exportable Reports**: Generation of complete diagnostic reports
* **Medical Validation**: Professional validation reminders

## 📋 System Requirements

### 🧩 Main Dependencies

* **Python 3.10+**
* **TensorFlow 2.15.0** and **Keras 3.0**
* **Streamlit 1.32.0**
* **NumPy**, **Pandas**, **Plotly**
* **OpenCV**, **Pillow**
* **Scikit-learn**
* **Matplotlib** and **Seaborn**


## Installation

1. **Clone the repository:**

```bash
git clone https://github.com/tavoofg/dermosan.git
cd dermosan
```

2. **Create a virtual environment:**

```bash
python -m venv dermosan_env
# Windows
dermosan_env\Scripts\activate
# Linux/Mac
source dermosan_env/bin/activate
```

3. **Install dependencies:**

```bash
pip install -r requirements.txt
```

4. **Verify file structure:**

```
dermosan/
├── app.py                   # Main application (Streamlit interface)
├── requirements.txt         # Environment dependencies
├── src/                     # System modules
│   ├── config.py            # General system configuration
│   ├── predictor.py         # Prediction module
│   └── utils.py             # Interface utilities
├── Trained_Model/           # 📁 The trained model (.h5) will be placed here after training
│   └── best_resnet152.h5    # Final ResNet152 model
├── Training_Code/           # Scripts and data for model training
│   ├── train_resnet152.py   # Model training script
│   └── IMG_CLASSES/         # 📁 Dataset from Kaggle goes here:
│                            # https://www.kaggle.com/datasets/ismailpromus/skin-diseases-image-dataset
└── verify_model.py          # Model validation script

```

## System Usage

### Run the application:

```bash
streamlit run app.py
```

### Access the interface:

* Open your browser at: `http://localhost:8501`
* Upload a dermatological image (JPG, PNG)
* Review the image quality analysis
* Obtain diagnosis and recommendations
* Generate a medical report

## Model Performance

* **Architecture**: ResNet152 with *transfer learning*
* **Estimated accuracy**: ~95%
* **Dataset**: 27,000+ dermatological images
* **Validation**: 80/10/10 split (train/val/test)
* **Optimizations**: *Data augmentation*, *class balancing*


## Code Structure

### `app.py`

Main **Streamlit** application with:

* Full user interface
* Image upload handling
* Results visualization
* Exportable report generation

### `src/predictor.py`

Prediction module that includes:

* Loading and management of the trained model
* Preprocessing of dermatological images
* Prediction generation
* Automatic image quality analysis

### `src/config.py`

Centralized configuration:

* Model parameters
* Medical information for each disease
* Confidence thresholds
* General application settings

### `src/utils.py`

Interface utilities:

* **Streamlit** visual components
* Interactive charts and visualizations
* Clinical result formatting
* Medical report export

## Clinical Use

### Recommendations:

1. **Use as a support tool** – It does not replace professional medical judgment.
2. **Professional validation** – Always confirm with a certified dermatologist.
3. **Image quality** – Use clear, well-lit photographs.
4. **Urgent cases** – Immediate attention for melanomas or suspicious lesions.

### Limitations:

* Does not diagnose all dermatological conditions.
* Requires validation by a medical professional.
* Performance depends on image quality.
* Does not replace biopsy or other confirmatory studies.

## Model Training

The model was trained using:

* **Base**: ResNet152 pretrained on *ImageNet*
* **Fine-tuning**: Last 50 trainable layers
* **Optimizer**: Adam with adaptive *learning rate*
* **Augmentation**: Rotation, zoom, horizontal flip
* **Balancing**: Automatic class weights for imbalanced dataset

## Future Improvements

* [ ] Mobile version for medical devices
* [ ] Inclusion of more dermatological classes
* [ ] Simultaneous analysis of multiple lesions
* [ ] Temporal tracking of clinical cases
* [ ] REST API for integration with other systems


## Development Team

Developed within the framework of the research project **“DERMOSAN – Artificial Intelligence-Assisted Dermatological Diagnostic System”**, as part of the innovation initiatives promoted by the **Universidad Nacional de Cañete (UNDC)**.

**Team members:**

* **Gustavo Fernández-Gutiérrez** – Lead developer and principal researcher
* **Andry Diego-Calagua** – Developer and associate researcher
* **Alex Pacheco-Pumaleque** – Academic advisor

**Funding entity:** Directorate of Innovation and Technology Transfer (DITT) – Universidad Nacional de Cañete (UNDC)


## License

This project is distributed under the **Attribution 4.0 International (CC BY 4.0)** license.
[https://opensource.org/license/MIT](https://opensource.org/license/MIT)

**DERMOSAN – UNDC 2025** was developed for research and medical application purposes, aimed at supporting dermatological diagnosis through artificial intelligence.


## 💰 Grant Information

This work was funded by the **Directorate of Innovation and Technology Transfer (DITT)** of the **Vice Presidency for Research** at the **Universidad Nacional de Cañete (UNDC)**, within the framework of the **“Research Competition for the Development of Innovations and Intellectual Property”**, under **contract number 017-2024**.

## Medical Disclaimer

This system is a **diagnostic support tool** and **does not replace**:

* Professional clinical judgment
* In-person medical evaluation
* Necessary complementary studies
* Biopsy when indicated

Always consult a **certified dermatologist** for diagnostic confirmation and appropriate treatment.

---

**Dermosan v1.0.0** – Automated Dermatological Diagnostic System
*DERMOSAN – UNDC 2025*
