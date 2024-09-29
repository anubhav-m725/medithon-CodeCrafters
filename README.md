Breast cancer detection using Machine Learning

Frontend:
User uploads an image.
The image is displayed in a preview, and the "Analyze Image" button becomes clickable.
On clicking the button, the frontend makes a request to the Flask API for analysis.

Backend:
Flask receives the image and processes it to extract features.
The features are fed into the pre-trained Logistic Regression model.
The model returns a result (either "cancer detected" or "no cancer detected") with a confidence score.
Flask sends this result back to the frontend.

Result Display:
The frontend receives the result and displays it to the user (either a green success message for no cancer or a red warning for potential cancer)
