🐱🐶 Cat vs Dog Image Classification

This project implements multiple Convolutional Neural Network (CNN) architectures to classify images of cats and dogs. The models range from custom CNNs (from scratch, LeNet, AlexNet) to pretrained architectures (VGG16, ResNet50, DenseNet121).

The dataset is organized in directories for each class, and the models are trained and evaluated on train, validation, and test splits.

🛠 Features

Load and preprocess images from directories.

Normalize images and split into training, validation, and test sets.

Implement multiple CNN architectures:

Custom CNN (from scratch)

LeNet

AlexNet

VGG16 (pretrained)

ResNet50 (pretrained)

DenseNet121 (pretrained)

Evaluate models using accuracy and classification report.

Compare performances of different architectures.

⚡ Installation

Install the required Python packages:

pip install tensorflow scikit-learn numpy opencv-python


Optional: Use GPU acceleration with TensorFlow if available for faster training.

📝 Usage

Organize your dataset in the following structure:

cat_dog_data/
└─ Train/
   ├─ Cat/
   │  ├─ cat1.jpg
   │  ├─ ...
   └─ Dog/
      ├─ dog1.jpg
      ├─ ...


Set the dataset path in your script:

train_path = '/content/cat_dog_data/Train'


Run the Python script to train and evaluate all models:

python cat_dog_classification.py


Check the printed accuracy for each model on the test set.

🔧 How It Works

Data Loading: Images are loaded from directories, resized to (224, 224), and labeled according to folder names.

Data Splitting: Split into train, validation, and test sets.

Normalization: Images are scaled to [0, 1].

Modeling: Multiple CNN architectures are created and compiled:

Custom CNN, LeNet, AlexNet

Transfer learning with VGG16, ResNet50, DenseNet121 (pretrained on ImageNet)

Training: Each model is trained for 10 epochs with batch size 64.

Evaluation: Test set accuracy is computed, predictions are thresholded at 0.5.

📂 Project Structure
cat-dog-classification/
│
├─ cat_dog_classification.py    # Main script
├─ README.md                    # Documentation



🧠 Example Output
Custom CNN Accuracy: 0.87
LeNet Accuracy: 0.84
AlexNet Accuracy: 0.88
VGG16 Accuracy: 0.90
ResNet50 Accuracy: 0.91
DenseNet121 Accuracy: 0.92

🔗 Dependencies

Python 3.x

TensorFlow / Keras

NumPy

OpenCV

Scikit-learn
