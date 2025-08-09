## Cats Vs Dogs Classification using CNN

_This project aims to classify images of cats and dogs using a Convolutional Neural Network (CNN). The model is trained on the Dogs vs Cats dataset from Kaggle and applies convolutional, pooling, and dense layers to learn image features for binary classification._

#### Dataset Details
- Dataset: Cats Vs Dogs
- Source: Kaggle (https://www.kaggle.com/datasets/salader/dogs-vs-cats )
- Training Samples: 2500 images each
- Image Shape: Resized to 256×256 pixels  
- Labels: Integers from 0 to 9 (10 classes)

#### ML Workflow: 
1. Importing Libraries
    1. `TensorFlow`, `Keras` for model building and training
    2. `NumPy`, `Pandas` for data handling
    3. `Matplotlib` for visualization
2. Data Loading & Preprocessing
    1. Images loaded from the dataset folder
    2. Normalized pixel values to range [0,1]
    3. Labels assigned for cats (0) and dogs (1)
3. Model Architecture
    1. Conv2D Layers with ReLU activation for feature extraction
    2. Batch Normalization to stabilize learning
    3. MaxPooling2D to reduce spatial dimensions
    4. Dense Layers for classification
    5. Dropout to prevent overfitting
    6. Sigmoid Output Layer for binary classification  
    >[Conv2D → BatchNorm → MaxPool] × 3  
    >Flatten  
    >Dense (128) → Dropout  
    >Dense (64) → Dropout  
    >Dense (1, sigmoid)  
    7. Loss Function: Binary Crossentropy
    8. Optimizer: Adam
4.  Model Training
    1. Compiled and trained the model for 10 epochs. 
5. Model Optimization
    1. Batch Normalization used for faster convergence
    2. Dropout applied to reduce overfitting
6. Model Evaluation
    1. Metrics used: `Accuracy`

#### Results
Accuracy Score: 0.96

Loss: 0.10

Val Accuracy: 0.83

Val Loss: 0.57


#### Improvements
1. Using Data augmentation will result in better generalization
2. Hyperparameter tuning could be done for higher accuracy

#### Visualizations
Accuracy graph before Optimization 

<img width="724" height="522" alt="Screenshot 2025-08-09 183900" src="https://github.com/user-attachments/assets/285eeec7-4743-4203-ad98-23ef0cb1b02e" />

Accuracy graph after Optimization

<img width="724" height="522" alt="Screenshot 2025-08-09 183907" src="https://github.com/user-attachments/assets/e1b98d0e-36d6-4292-9279-7b96d65dbbb2" />

Loss graph before Optimization

<img width="724" height="522" alt="Screenshot 2025-08-09 183914" src="https://github.com/user-attachments/assets/af33806b-407b-4a1b-b868-8265db18aff4" />

Loss graph after Optimization

<img width="724" height="522" alt="Screenshot 2025-08-09 183922" src="https://github.com/user-attachments/assets/b8317fd7-671f-44c5-af90-0e32a29141da" />

#### Assumptions
1. Images are resized to 256×256 without losing critical features.   
