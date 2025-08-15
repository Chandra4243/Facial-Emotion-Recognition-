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
EPOCHS = 15
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
Chandrabhushan Upadhyay

Contact: chandr4243@gmail.com

LinkedIn: www.linkedin.com/in/chandr34
