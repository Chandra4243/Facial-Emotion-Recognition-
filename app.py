# app.py
# --- Imports ---
import torch
import torch.nn as nn
import torchvision.transforms as T
from flask import Flask, request, jsonify, render_template
from PIL import Image
import io
import timm # Make sure timm is installed: pip install timm

# --- Model Definition ---
# This class must be defined to load the model's structure correctly.
# It's identical to the one in your Colab notebook.
class FaceModel(nn.Module):
    def __init__(self):
        super(FaceModel, self).__init__()
        # Using efficientnet_b1 as in the notebook, with 7 output classes
        self.eff_net = timm.create_model('efficientnet_b4', pretrained=False, num_classes=7)

    def forward(self, images, labels=None):
        logits = self.eff_net(images)
        if labels is not None:
            loss = nn.functional.cross_entropy(logits, labels)
            return loss, logits
        return logits

# --- App Initialization ---
app = Flask(__name__)

# --- Load Model ---
# Make sure 'best_model.pt' is in the same directory as this script.
try:
    model = FaceModel()
    # Load the state dictionary. map_location ensures it loads on CPU if no GPU is available.
    model.load_state_dict(torch.load('best_model.pt', map_location=torch.device('cpu')))
    model.eval() # Set the model to evaluation mode
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

# --- Image Transformations ---
# Define the same transformations used during your model's validation.
# The model expects a 3-channel, 48x48 image tensor.
transform = T.Compose([
    T.Resize((48, 48)),
    T.ToTensor(),
    # Although the images are grayscale, the model was likely trained on 3-channel
    # tensors, so we normalize for 3 channels.
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # Normalizing for 3 channels
])

# --- Class Labels ---
# The order must match the training folder structure.
class_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# --- Routes ---
@app.route('/')
def home():
    """Renders the home page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handles the prediction request."""
    if model is None:
        return jsonify({'error': 'Model is not loaded'}), 500
        
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected for uploading'}), 400

    try:
        # Read image bytes
        img_bytes = file.read()
        image = Image.open(io.BytesIO(img_bytes))
        
        # Convert image to RGB to ensure 3 channels
        image = image.convert("RGB")

        # Preprocess the image
        image_tensor = transform(image).unsqueeze(0) # Add batch dimension
        
        # Make a prediction
        with torch.no_grad():
            logits = model(image_tensor)
            # Get probabilities using softmax
            probabilities = torch.nn.functional.softmax(logits, dim=1)[0]
            # Get the top class index
            top_class_index = probabilities.argmax().item()
            prediction = class_names[top_class_index]
            confidence = probabilities[top_class_index].item()

        # Return the prediction and confidence
        return jsonify({
            'prediction': prediction.capitalize(),
            'confidence': f"{confidence:.2%}"
        })

    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({'error': f'An error occurred during prediction: {e}'}), 500

# --- Main ---
if __name__ == '__main__':
    # Use a production-ready server like Gunicorn or Waitress in a real deployment
    app.run(debug=True, host='0.0.0.0', port=5000)
