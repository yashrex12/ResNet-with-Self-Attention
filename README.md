Image classification model based on Resnet but with added features such as Self Attention and regularization techniques to prevent overfitting during testing.

Files in the repository:
selfattention.py - Self Attention module for increased accuracy
resnet.py - ResNet based model with 2 Convolutional layers due to computational constraint but can be added more if necessary. Integrated self attention and dropout for better generalization
ResidualBlock.py - Residul blocks to be used in the network 
All of these files are customizable depending on your task and computational capabalities.

model.py - training and testing of the model. 
Trained on CIFAR-10 Dataset. 
Accuracy of the network on the training set: 90 % 
Accuracy of the network on the validation set: 94 % 
Best model saved with accuracy: 94.93%

Accuracy of the best model on the test images: 72.35

Thank You!