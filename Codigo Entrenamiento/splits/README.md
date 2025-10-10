# 📁 Carpeta: splits

Esta carpeta contiene los archivos de división del dataset (*splits*) utilizados durante el entrenamiento, validación y prueba del modelo.

## 📄 Contenido
- **split_train.csv** → Lista de imágenes y etiquetas utilizadas para el entrenamiento.  
- **split_val.csv** → Lista de imágenes y etiquetas empleadas para validación.  
- **split_test.csv** → Lista de imágenes y etiquetas destinadas a la fase de prueba.  

Cada archivo CSV incluye las columnas:
- `filename` → Nombre del archivo de imagen  
- `label` → Clase o categoría dermatológica correspondiente  

## 🔍 Descripción general
Estos archivos permiten reproducir los experimentos manteniendo la misma proporción (80/10/10) utilizada en el estudio original.  
El conjunto de datos base corresponde al *Skin Diseases Image Dataset* (Hossain, 2021) disponible en Kaggle.  

> ⚠️ No modificar estos archivos si se desea conservar la reproducibilidad del entrenamiento.
