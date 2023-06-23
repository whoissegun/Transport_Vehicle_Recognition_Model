# Transport Vehicle Recognition Model

This project presents a model that uses deep learning to classify images of transport vehicles. It is implemented in Python using PyTorch, a popular open-source machine learning library.

The model is trained and evaluated on a multi-class image dataset that contains images of different types of transport vehicles, specifically airplanes, ships and cars. 

## Requirements
This project was developed on Google Colab and uses the following libraries:
- Python
- PyTorch
- Scikit-learn
- Matplotlib
- torchvision
- Google Colab's drive and file modules for accessing and managing files.

## Data
The dataset is downloaded from Kaggle via the Kaggle API and unzipped for use in the project. The dataset contains labeled images of Airplanes, Cars, and Ships that are split into training and testing sets. 

## Data Preprocessing
The images in the dataset are transformed before use. For the training set, the images are resized to 224x224 pixels and randomly rotated by 90 degrees. The images are then converted to tensors. For the test set, the images are simply resized to 224x224 pixels and converted to tensors.

## Model Architecture
The `Transport_Vehicle_Recognition` class represents the model used in this project. It is a multi-layer feed-forward neural network model that includes five linear layers, interspersed with batch normalization and ReLU activation function.

## Training the Model
The model is trained for 150 epochs. At each epoch, the model learns by adjusting its parameters to minimize the difference between its predictions and actual results. This difference is computed using the cross-entropy loss function. The Adam optimizer is used to update the model parameters based on the computed gradients.

## Evaluating the Model
The model is evaluated after every five training epochs on the test set. The model’s parameters that give the highest accuracy are considered the best and are saved for later use. The losses per epoch during training and testing are also plotted for visualization.

## Output
At the end of the training and evaluation, the model’s parameters with the best accuracy are loaded back into the model and saved. Also, a classification report showing precision, recall, f1-score, and support for each class is printed.

## How to Use
This model can be used as a starting point for any image classification task. Simply replace the dataset download link with the link to your own dataset on Kaggle. You may also need to adjust the image transformations and the model architecture to suit your specific task.
