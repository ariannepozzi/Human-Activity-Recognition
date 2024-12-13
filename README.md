# Human Activity Recognition (HAR) using Deep Learning

## Introduction
Human Activity Recognition (HAR) is a field of study that focuses on identifying physical activities such as walking, sitting, or running based on sensor data. In this project, we utilize deep learning to classify six activities using sensor data collected from smartphones equipped with accelerometers and gyroscopes.

The dataset contains over 10,000 samples, each with 561 features extracted from time-domain and frequency-domain signals. The activities to be classified are:
- Walking
- Walking Upstairs
- Walking Downstairs
- Sitting
- Standing
- Laying

## Data Overview
The dataset used for this project is a collection of sensor data from smartphones worn by 30 subjects performing six different activities. The key features of the dataset include:
- 10,299 samples
- 561 features per sample, extracted from accelerometer and gyroscope readings
- Six labeled activities: Walking, Walking Upstairs, Walking Downstairs, Sitting, Standing, and Laying

### Data Processing
- **Normalization**: All feature values were scaled to a range of 0 to 1 to ensure faster convergence during model training.
- **One-Hot Encoding**: Labels for the six activities were converted into one-hot vectors for multi-class classification.
- **Reshaping**: The 561 features were reshaped into a 561x1 format to make the data compatible with the convolutional neural network (CNN) model.

### Model Architecture
The CNN model architecture consists of the following:
1. **Convolutional Layers**: Two convolutional layers, each followed by max-pooling for dimensionality reduction.
2. **Fully Connected Layers**: Flattening operation followed by fully connected layers for final classification.
3. **Activation Functions**: ReLU activations were used to introduce non-linearity, while a softmax activation was applied in the final layer for multi-class classification.

### Training
- **Data Split**: The dataset was split into training (80%) and validation (20%) sets.
- **Training Parameters**: A batch size of 64 and 20 epochs were used to balance training time and performance.
- **Optimizer**: The Adam optimizer was employed for adaptive learning rates.
- **Loss Function**: Categorical cross-entropy was chosen as the loss function.

The model achieved near-perfect accuracy on the validation set, with training and validation losses converging efficiently over the epochs.

### Evaluation: Confusion Matrix
The confusion matrix provides a visual representation of the model's performance. For example:
- **Walking Upstairs**: All 270 samples were correctly classified.
- **Walking Downstairs**: 282 samples were correctly classified, with only 2 misclassified samples.
- **Standing**: 350 samples were correctly classified, with 4 samples misclassified as sitting.

### Performance Metrics
- **Training Curves**: Both training and validation losses decreased steadily during the initial epochs, with a plateau observed after epoch 6, indicating efficient convergence.
- **Accuracy Curves**: Training and validation accuracies increased rapidly in the first few epochs, plateauing at around 99% accuracy, demonstrating consistent performance across training and validation data.

### Comparison with Other Models
- **Naive Bayes**: The Naive Bayes classifier performed poorly, with an accuracy of 75%, due to its assumption of feature independence, which is unrealistic for sensor data. Unlike CNN, Naive Bayes does not perform feature extraction or account for sequential relationships in the data.
- **CNN**: The CNN model outperformed Naive Bayes, achieving near-perfect precision, recall, and F1-scores for all activities, with minor variations.

### Real-Time Predictions
This project also includes a demonstration of real-time predictions. The model is capable of classifying incoming sensor data in real time, simulating a scenario where sensor readings are classified continuously. A loop was set up to simulate predictions for 10 time steps, showcasing the model's ability to make predictions dynamically.

