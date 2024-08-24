# skin-Cancer-Detection
This project is a web-based application for detecting skin cancer using a deep learning model. It allows users to upload images of skin lesions and provides a prediction on the likelihood of the image representing a cancerous lesion.

# Features
Model Loading or Training: Automatically loads a pre-trained model if available, or trains a new model from scratch using the provided dataset.
Deep Learning Architecture: The model is built using a Convolutional Neural Network (CNN) with multiple layers to accurately classify skin cancer images.
Flask Web Interface: The application provides a simple and interactive web interface for uploading images and displaying predictions.
Image Preprocessing: Uploaded images are automatically resized and normalized before being fed into the model for prediction.
Model Persistence: Trained models are saved for future use, avoiding the need to retrain the model every time the application is run.
Files
app.py: The main Flask application file that handles image uploads, model loading/training, and predictions.
model.h5: The saved model file (if available) that the application loads to make predictions.
templates/: Contains the HTML templates for the web pages (index.html for uploads and result.html for displaying predictions).
# Installation
Clone the repository:

git clone https://github.com/yourusername/skin-cancer-detection.git
cd skin-cancer-detection
Install the required dependencies:

pip install -r requirements.txt
Prepare your dataset:

Ensure your training and testing data are organized in datasets/train and datasets/test directories, respectively. The images should be categorized into subfolders for each class (e.g., 'cancerous' and 'non-cancerous').
Run the application:


python app.py
Access the web application:

Open your browser and go to http://127.0.0.1:5000/.

Usage
Upload an Image: Use the web interface to upload an image of a skin lesion.
View Prediction: The application will preprocess the image, run it through the model, and display the likelihood that the lesion is cancerous as a percentage.
Model Training: If the saved model is not found, the application will automatically train a new model using the images in the datasets/train directory.
Model Architecture
The Convolutional Neural Network (CNN) used in this project consists of:

Conv2D Layers: Three convolutional layers with 32, 64, and 128 filters respectively, each followed by a MaxPooling layer to reduce the spatial dimensions.
Dense Layers: A fully connected layer with 128 units, followed by a sigmoid output layer for binary classification.
Activation Functions: ReLU is used for the hidden layers, and sigmoid is used for the output layer.
The model is trained using the Adam optimizer and binary cross-entropy loss function, with accuracy as the evaluation metric.

Contributing
Contributions are welcome! Feel free to submit issues or pull requests to help improve the project.

License
This project is licensed under the MIT License - see the LICENSE file for details
