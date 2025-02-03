# -ORANGE-speech-emotion-recognition-SER-MLP-SVM-KNN-
# Speech Emotion Recognition (SER) using MLP, SVM, and KNN

## Introduction
This project focuses on Speech Emotion Recognition (SER) using the RAVDESS dataset. The objective is to classify speech audio into eight emotion categories: **anger, sad, happy, neutral, calm, fearful, disgust, and surprised**. The project extracts various audio features and implements multiple classification models, including a **Multi-Layer Perceptron (MLP), Support Vector Machine (SVM), and K-Nearest Neighbors (KNN)**, to evaluate and compare their performance.

## Dataset
The dataset used is the **RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)**, which consists of 1440 speech audio files labeled with different emotions. Each file follows a specific naming convention to indicate the actor, emotion type, intensity, and other details.

### Filename Structure
Each audio file follows this pattern:
```
03-01-06-01-02-01-12.wav
```
Where:
- `03` - Modality (Audio-only)
- `01` - Vocal channel (Speech)
- `06` - Emotion (Fearful)
- `01` - Emotional intensity (Normal)
- `02` - Statement ("Dogs are sitting by the door")
- `01` - Repetition
- `12` - Actor ID (Even = Female, Odd = Male)

## Feature Extraction
The following features were extracted from the audio files using **Librosa**:
- **Mel-Frequency Cepstral Coefficients (MFCC)** (13 features)
- **Chromogram** (12 features)
- **Mel-scaled Spectrogram** (100 features)
- **Spectral Contrast** (7 features)
- **Tonnetz (Tonal Centroid Features)** (6 features)

## Data Preprocessing
- **Balancing the Dataset**: Ensured equal representation of emotions by resampling.
- **Feature Standardization**: Used `StandardScaler` to normalize extracted features.
- **Label Encoding**: Converted emotion labels into numeric form using `LabelEncoder`.

## Model Implementation
### 1. Multi-Layer Perceptron (MLP) Classifier
- **Activation Function**: ReLU
- **Optimizer**: Adam
- **Fine-tuned Parameters**: Adjusted learning rate, hidden layers, and epochs

### 2. Support Vector Machine (SVM)
- **Kernel**: RBF
- **Regularization (C)**: Tuned for optimal performance

### 3. K-Nearest Neighbors (KNN)
- **Number of Neighbors (K)**: Experimented with different values for optimal accuracy
- **Distance Metric**: Euclidean

## Model Evaluation
The models were evaluated based on:
- **Accuracy**
- **Precision, Recall, and F1-score**
- **Confusion Matrix**
- **K-Fold Cross-Validation**

## Results and Comparison
A comparative analysis was performed between MLP, SVM, and KNN classifiers. MLP achieved the highest performance due to its ability to capture complex speech patterns.

## Installation and Dependencies
To run the project, install the following dependencies:
```bash
pip install numpy pandas librosa scikit-learn matplotlib
```

## Running the Project
1. Clone the repository:
```bash
git clone https://github.com/your-repo/speech-emotion-recognition.git
```
2. Navigate to the project folder:
```bash
cd speech-emotion-recognition
```
3. Run the Jupyter Notebook or Python script to extract features, train, and evaluate models:
```bash
jupyter notebook SRN_SECTION.ipynb
```

## Conclusion
This project successfully classifies speech emotions using machine learning techniques. The MLP model performed the best among the three classifiers. Future improvements can include deep learning models such as CNNs or LSTMs to enhance performance.

## Acknowledgments
- **Dataset**: RAVDESS Dataset
- **Libraries Used**: Librosa, Scikit-learn, NumPy, Matplotlib

