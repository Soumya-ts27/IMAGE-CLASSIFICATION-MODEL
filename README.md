# IMAGE-CLASSIFICATION-MODEL

**COMPANY**: CODETECH IT SOLUTIONS

**NAME**: TARIMELA SRINIVASA SOUMYA

**INTERN ID**:CT12WJVV

**DOMAIN**: MACHINE LEARNING

**BATCH DURATION**: JANUARY 5th,2025 to APRIL 5th,2025

**MENTOR NAME**: NEELA SANTHOSH

This Python program implements a Convolutional Neural Network (CNN) using TensorFlow and Keras to classify images from the CIFAR-10 dataset. The CIFAR-10 dataset consists of 60,000 32x32 color images categorized into 10 classes, including objects like airplanes, cars, and animals. The program follows a structured approach, including data loading, preprocessing, model building, training, evaluation, and visualization.

---

## Step 1: Importing Libraries

The required libraries are imported at the beginning:
- **TensorFlow and Keras** for building and training the CNN model.
- **Matplotlib** for visualizing the training history.
- **NumPy** for numerical operations and array manipulations.

---

## Step 2: Loading the Dataset

The CIFAR-10 dataset is loaded using the `keras.datasets.cifar10.load_data()` function. This function splits the dataset into:
- `x_train` and `y_train`: Training images and their corresponding labels.
- `x_test` and `y_test`: Test images and labels for evaluating the model.

Each image is a 32x32 RGB image with three color channels (Red, Green, Blue). The labels are integers ranging from 0 to 9, representing the 10 classes.

---

## Step 3: Data Preprocessing

To optimize the neural network's performance, the pixel values of the images are normalized by scaling them between **0 and 1** using:

```python
x_train, x_test = x_train / 255.0, x_test / 255.0
```

Normalization helps accelerate training by ensuring that all input values are within a uniform range.

---

## Step 4: Defining Class Names

A list of class names is defined to represent each label in a human-readable format:

```python
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
```

These class names will help interpret the model's predictions.

---

## Step 5: Building the CNN Model

The program constructs a sequential CNN using `keras.Sequential()`, which consists of multiple layers:

1. **Convolutional Layer 1**  
    - `Conv2D` with 32 filters of size **3x3** and ReLU activation.
    - Extracts features like edges and textures.

2. **MaxPooling Layer 1**  
    - Reduces feature map size using **2x2** pooling to minimize computation.

3. **Convolutional Layer 2**  
    - `Conv2D` with 64 filters and ReLU activation for deeper feature extraction.

4. **MaxPooling Layer 2**  
    - Further down-sampling using **2x2** pooling.

5. **Convolutional Layer 3**  
    - `Conv2D` with 128 filters and ReLU activation to learn complex features.

6. **Flatten Layer**  
    - Converts the 2D feature maps into a 1D vector.

7. **Dense Layer 1**  
    - Fully connected layer with 128 neurons and ReLU activation.

8. **Output Layer**  
    - 10 neurons representing each class with **Softmax** activation to output class probabilities.

---

## Step 6: Compiling the Model

The model is compiled using:
- **Adam Optimizer**: Adaptive learning rate optimization for faster convergence.
- **Sparse Categorical Crossentropy Loss**: Suitable for multi-class classification with integer labels.
- **Accuracy Metric**: Tracks how many predictions match the actual labels.

---

## Step 7: Training the Model

The model is trained using the `fit()` function:
- **Epochs = 10**: The training loop runs 10 times through the entire dataset.
- **Validation Data**: The test set is used for validation during training to monitor accuracy.

The training history is stored in the `history` object, which contains data on accuracy and loss over epochs.

---

## Step 8: Evaluating the Model

After training, the model is evaluated on the test set using `model.evaluate()`. The test accuracy is printed to assess the model's generalization capability.

---

## Step 9: Visualizing Training Results

The program plots the training and validation accuracy using Matplotlib. This helps in analyzing:
- **Overfitting**: If the training accuracy is much higher than the test accuracy.
- **Underfitting**: If both accuracies are low, indicating the model is too simple.

---

## Conclusion

This CNN is an effective approach for image classification tasks like CIFAR-10. It leverages convolutional layers for hierarchical feature extraction, followed by dense layers for classification. With appropriate preprocessing, optimization, and evaluation, the model achieves good accuracy, demonstrating its ability to distinguish between various image classes. Additionally, visualizing the accuracy trends helps monitor model performance and make adjustments if necessary.

**OUTPUT**:![Image](https://github.com/user-attachments/assets/6cda3d12-b28f-457c-9cba-79c06427f94f)
