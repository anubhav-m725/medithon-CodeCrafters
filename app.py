from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import warnings
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Initialize Flask app
app = Flask(__name__)

# Load breast cancer dataset and train the model
breast_cancer_dataset = load_breast_cancer()

data_frame = pd.DataFrame(breast_cancer_dataset.data, columns=breast_cancer_dataset.feature_names)
data_frame['label'] = breast_cancer_dataset.target

X = data_frame.drop(columns='label', axis=1)
Y = data_frame['label']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

# Initialize the Logistic Regression model
model = LogisticRegression()
model.fit(X_train, Y_train)

# Suppress the warning for valid feature names
warnings.filterwarnings('ignore', message='X does not have valid feature names')

@app.route('/analyze', methods=['POST'])
def analyze_image():
    try:
        # Simulate feature extraction from image (here, we use a static input for testing)
        input_data = request.json.get('input_data')  # In production, replace this with actual image processing logic
        
        # Convert input data to numpy array and reshape for prediction
        input_data_as_numpy_array = np.asarray(input_data)
        input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

        # Predict using the model
        prediction = model.predict(input_data_reshaped)
        confidence = model.predict_proba(input_data_reshaped).max() * 100  # Confidence score

        # Return prediction result
        result = {
            'isCancerous': bool(prediction[0]),
            'confidence': round(confidence, 2)
        }
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
