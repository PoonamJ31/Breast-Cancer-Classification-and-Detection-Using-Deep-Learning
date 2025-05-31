# Breast Cancer Classification and Detection Using Deep Learning

This project was developed as the Final Year Bachelor of Engineering (B.E.) Project in the Computer Engineering Department at SITRC, Savitribai Phule Pune University. It presents a deep learning-based system for early and accurate classification of breast cancer tumors as benign or malignant using histopathology images.

🔬 **This work has been peer-reviewed and published** in the *Journal of Emerging Technologies and Innovative Research (JETIR)*, Volume 10, Issue 5 (May 2023).

---

## 📚 Research Publication
- **Title:** Breast Cancer Classification and Detection using Deep Learning  
- **Authors:** Poonam Jadhav, Sunidhi Jain, Kajal Patil, Komal Bhadane  
- **Published in:** *JETIR*, Volume 10, Issue 5, May 2023  
- **ISSN:** 2349-5162 | UGC Approved Journal No: 63975  
- **Paper ID:** JETIR2305636

---

## 🗂️ Project Structure

breast-cancer-detection/

├── data/                         # Contains the BreakHis dataset

├── notebooks/                   # Jupyter notebook for training + evaluation

├── breast_cancer_classification.ipynb

├── requirements.txt

└── README.md


---

## 🧪 Dataset
- **Name:** BreakHis v1 (Breast Histopathology Images)  
- **Source:** [Kaggle - ambarish/breakhis](https://www.kaggle.com/datasets/ambarish/breakhis)  
- **Classes:** Benign and Malignant  
- **Preprocessing:**
  - Resized images to 224x224
  - Normalized pixel values
  - 80/20 train-validation split
  - Flattened folder structure for model compatibility
 
---

## 📦 Dependencies
Install the required packages using:

pip install -r requirements.txt

🔧 Main Libraries

TensorFlow / Keras – For building and training deep learning models

scikit-learn – For traditional machine learning, preprocessing, and evaluation

OpenCV – For image processing and computer vision tasks

matplotlib / seaborn – For data visualization and plotting

numpy / pandas – For numerical operations and data manipulation

---

## ✨ Future Improvements
- Integrate transfer learning using models like ResNet50 or InceptionV3
- Deploy using Streamlit or Flask for real-time predictions
- Add support for direct image upload and classification
- Implement Grad-CAM for visual explanation of model predictions
