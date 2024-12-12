# Facial Emotion Recognition with Deep Learning

This project explores facial emotion recognition using deep learning techniques, specifically Convolutional Neural Networks (CNNs), applied to the FER-2013 dataset.  The goal is to classify facial expressions into seven basic emotions: Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral.

## Notebook Overview

The `final-ds-301.ipynb` notebook details the development and evaluation of the model. The process includes:

1. **Data Loading and Preprocessing:**  The FER-2013 dataset is loaded from a CSV file, and pixel data is cleaned and normalized.  The dataset is split into training, validation, and test sets based on the 'Usage' column in the dataset.

2. **Dataset Analysis:**  The notebook provides visualizations of sample images from each emotion class and a histogram illustrating the class distribution within the training set.  This highlights the inherent class imbalance in the dataset.

3. **Model Building and Tuning:**  Two main CNN architectures are experimented with:
    * **ReLU CNN:**  A CNN using ReLU activation functions is developed.  Hyperparameter tuning is performed using Keras Tuner's `RandomSearch` to find optimal configurations for filter sizes, number of filters, dropout rates, and L2 regularization.
    * **LeakyReLU CNN:** Another CNN using LeakyReLU activation functions is developed for comparison and robustness. Hyperband Tuning is used to find the best hyperparameters of this model.

4. **Model Training and Evaluation:**  The tuned models are trained using the training and validation sets. The model incorporates `ReduceLROnPlateau` for adaptive learning rate adjustment and `EarlyStopping` to prevent overfitting.  Model performance is evaluated on the test set using metrics like accuracy, precision, recall, and F1-score.

5. **Ensemble Stacking with Meta-Learner:** Predictions from both ReLU and LeakyReLU models are stacked and used to train a meta-learner (DecisionTreeClassifier).  Hyperparameter tuning for the Decision Tree is performed to optimize the ensemble model's performance. Final Accuracy is calculated using this Stacked model and is slightly better than both models.

6. **Visualization and Analysis:** Training and validation accuracy and loss are plotted to visualize the learning process. Confusion matrices and classification reports provide detailed insights into the model's performance on each emotion class.

## Challenges and Explorations

The notebook discusses several challenges encountered during development, including:

* **Computational Resources:** Training large pre-trained models was limited by available GPU and RAM.
* **Dataset Limitations:** The relatively small size and inherent noise/mislabeling within the FER-2013 dataset posed challenges for achieving higher accuracy.

Several approaches were explored to address these challenges:

* **Class Weights:**  Experimentation with class weights to handle class imbalance.
* **Data Augmentation:** Attempts at using data augmentation were made.
* **Noise Reduction:** Tried noise reduction techniques
* **Bagging:** Attempted ensemble methods like Bagging.
* **Different Meta-Learners:** Explored various classifiers for the meta-learner in the stacked ensemble.

## Running the Notebook

To run the notebook, you'll need the following:

* Python 3.10+
* TensorFlow/Keras
* Keras Tuner
* scikit-learn
* pandas
* NumPy
* matplotlib
* seaborn
* OpenCV (cv2)
* Albumentations


You'll also need to download the FER-2013 dataset and place it in the specified path within the notebook.
