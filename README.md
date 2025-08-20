Facial Expression Recognition 🎭
A web-based application that utilizes a deep learning model to recognize and classify human facial expressions from an uploaded image. The project combines a PyTorch backend, a Flask server, and a responsive front-end.

✨ Features \& Technology Stack
This project leverages a range of modern technologies to deliver a robust and interactive experience.

Interactive Web Interface: A clean, responsive UI built with Tailwind CSS, HTML, and JavaScript.

Client-Side Face Detection: Employs face-api.js in the browser to validate that an uploaded image contains a face before processing.

AI-Powered Emotion Prediction: The core of the application uses an EfficientNet-B4 model from PyTorch and timm to classify expressions into seven categories: Happy, Sad, Neutral, Angry, Disgust, Fear, and Surprise.

Confidence Score: Displays the prediction along with the model's confidence percentage.

Flask Backend: A lightweight Python and Flask backend serves the model and handles prediction requests.

📂 File Structure
The project is organized to be simple and easy to navigate.

.
├── app.py                      # Flask server \& prediction logic
├── best\_model.pt               # Trained PyTorch model weights
├── requirements.txt            # Python dependencies
├── templates/
│   └── index.html              # Frontend HTML, CSS, and JavaScript
└── Facial\_Expression\_Recognition\_with\_PyTorch\_(1).ipynb  # Notebook for model training

🚀 Getting Started
Follow these steps to set up and run the application on your local machine.

Prerequisites
Python 3.8 or newer

pip (Python package installer)

Installation \& Setup
Clone the Repository:

git clone https://github.com/your-username/facial-expression-recognition.git
cd facial-expression-recognition

Create and Activate a Virtual Environment:
This isolates the project's dependencies from your system's Python installation.

On Windows:

python -m venv venv
.\\venv\\Scripts\\activate

On macOS \& Linux:

python3 -m venv venv
source venv/bin/activate

Install Dependencies:

pip install -r requirements.txt

Run the Application:

python app.py

🧠 Model Development
The deep learning model was trained and developed in the Facial\_Expression\_Recognition\_with\_PyTorch\_.ipynb    Jupyter Notebook.

Dataset
The model is trained on the Face Expression Recognition Dataset, which contains 28,821 training images and 7,066 validation images. The dataset can be found on Kaggle.

Architecture
The model uses EfficientNet-B4, a pre-trained convolutional neural network from the timm library. Transfer learning was used to fine-tune the model for the 7 distinct expression classes.

Training
The training process, defined in the notebook, includes data augmentation techniques (like Random Flips and Rotations) to improve model robustness. The training loop runs for 40 epochs, and the best-performing model weights are saved as best\_model.pt based on the lowest validation loss on epoch number 18.

Inference
The notebook also includes a section demonstrating how to load the saved model and perform inference on new images, along with a function to visualize the prediction probabilities.

🖋️ Author

©️ Name: Chandrabhushan Upadhyay				LinkedIn: www.linkedin.com/in/chandr34			Email: chandr4243@gmail.com



=======


For Training Model


Facial Expression Recognition using PyTorch

This project builds a deep learning model to classify human facial expressions into one of seven categories: angry, disgust, fear, happy, neutral, sad, and surprise. It leverages transfer learning with a pre-trained EfficientNet-B1 model and is implemented in PyTorch. The notebook covers the complete workflow from data loading and augmentation to training, validation, and inference on a new image.


Note - The code will only run on GPU, either it should be in the device or add GPU as run time to T4 in google collab 


Features
          
  Recognizes 7 distinct facial expressions.
          
  Utilizes a pre-trained EfficientNet-B1 model for high performance.

  Includes data augmentation techniques (Random Flips, Rotations) to improve model robustness.

  Complete training and validation loop with progress tracking using tqdm.

  Saves the best-performing model based on validation loss.

  Provides a function for easy inference and visualization of results.

Technologies Used

  Python

  PyTorch: The core deep learning framework.

  timm (PyTorch Image Models): Used for accessing pre-trained models like EfficientNet.

  torchvision: For data transformations and loading.

  NumPy: For numerical operations.

  Matplotlib: For visualizing images and results.

Dataset

The model is trained on the Face Expression Recognition Dataset.

Source: Available on Kaggle https://www.kaggle.com/datasets/jonathanoheix/face-expression-recognition-dataset

The dataset contains 28,821 training images and 7,066 validation images.

Classes: Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise.

Setup and Usage
1. Installation

Clone the dataset and install the required Python libraries:
!git clone https://github.com/parth1620/Facial-Expression-Dataset.git


# Install necessary packages
pip install torch torchvision timm
pip install --upgrade opencv-contrib-python numpy matplotlib tqdm
2. Training the Model
To train the model, simply run the cells in the Jupyter Notebook sequentially. The key training parameters are defined in the "Configurations" section:

Python

LR = 0.001
BATCH_SIZE = 32
EPOCHS = 40
DEVICE ='cuda'
MODEL_NAME = 'efficientnet_b1'
The training loop will run for 15 epochs, and the best model weights will be saved as best_model.pt based on the lowest validation loss.

3. Inference
The final section of the notebook demonstrates how to load the saved best_model.pt and perform inference on a single image from the validation set. The view_classify function provides a clear visualization of the model's prediction probabilities.

Model Architecture
The model uses EfficientNet-B1, a powerful and efficient convolutional neural network, with weights pre-trained on the ImageNet dataset. This transfer learning approach allows the model to achieve high accuracy with less training time. The final classifier layer was replaced with a new one tailored for the 7 expression classes in our dataset.

Results
The model was trained for 15 epochs, achieving a validation accuracy of approximately 69.2%. The training progress, including loss and accuracy per epoch, is logged in the notebook output. The saved best_model.pt corresponds to the epoch with the lowest validation loss.


Author
©️ Chandrabhushan Upadhyay

Contact: chandr4243@gmail.com

LinkedIn: www.linkedin.com/in/chandr34
