F%FFacial Expression Recognition 🎭
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



