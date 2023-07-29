# Breast Cancer Classification using DenseNet201

This project uses a deep learning approach to classify breast cancer images into benign or malignant categories. The model is built and trained using the Keras library in Python and uses the DenseNet201 architecture for feature extraction.

## Libraries Used
The following Python libraries are used in this project:

- numpy
- pandas
- matplotlib
- tensorflow
- keras
- sklearn
- PIL
- cv2
- os
- json
- scipy
- tqdm
- gc
- functools
- collections
- itertools
- pickle

## Dataset

The dataset consists of images of breast cancer cells, which are classified as either benign or malignant. These images are loaded and preprocessed before being used to train the model.

## Preprocessing

The preprocessing steps include:

- Loading the images from a specified directory and resizing them.
- Assigning labels to the images (0 for benign, 1 for malignant).
- Shuffling the data.
- Splitting the data into training and validation sets.

Data augmentation techniques such as zooming, rotation, and flipping are applied to the training data to increase its diversity and variability.

## Model Building and Training

The DenseNet201 architecture, pre-trained on the ImageNet dataset, is used as the backbone of the model. Additional layers are added to the model, including GlobalAveragePooling2D, Dropout, BatchNormalization, and Dense layers.

The model is compiled with a binary cross-entropy loss function and the Adam optimizer. The learning rate is reduced if no improvement in validation accuracy is observed after a certain number of epochs.

## Evaluation and Visualization

The model's performance is evaluated using accuracy as the metric. The training and validation losses and accuracies are plotted for each epoch to visualize the model's learning process.

## Saving the Model

The trained model and the image data generator are saved for future use. The model is saved as an HDF5 file, and the image data generator is saved as a pickle file.

---

Please replace `<pathToLocation>` with the actual path where your data is stored or where you want to save your files.
