Breast Cancer Prediction using Machine Learning

This repository contains a machine learning project that uses the Random Forest Classifier to predict breast cancer diagnosis based on features extracted from breast tissue. The dataset used is the Breast Cancer Data.


Key Points:
Random Forest Model Training: The model is trained on the dataset and saved in memory when the Flask app runs.
Form Data Handling: The predict function collects the form inputs, scales them, and predicts whether the tumor is malignant or benign.
Prediction Output: The result is displayed on a separate page (e.g., result.html) with the prediction and confidence score.



File Structure: 
├── templates
│   ├── index.html             # Web Interface
│   ├── tumor.html             # Tumor explanation section
│   ├── footer.html            # Footer template
│   └── copyrights.html        # Copyrights section
├── static
│   └── style.css              # Custom styles for the web interface
├── app.py                     # Flask backend
├── Breast Cancer Data.csv      # Dataset
├── random_forest.py            # Model training and prediction logic
└── README.md                  # Project documentation




