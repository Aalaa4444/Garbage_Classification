# Garbage Classification with Feature Selection via Genetic Algorithm

## Project Overview

This project implements a robust pipeline for image-based garbage classification. It leverages transfer learning for feature extraction, followed by an advanced feature selection technique using a Genetic Algorithm (GA) to optimize a Support Vector Machine (SVM) classifier. The goal is to enhance model performance and generalization by identifying the most discriminative subset of features from pre-trained convolutional neural networks.

## Methodology

The implemented solution follows a multi-stage, hybrid approach:

### 1. Data Preparation & Augmentation
- The dataset is partitioned into dedicated **training** and **validation** sets.
- To combat overfitting and improve model generalization, we employ **data augmentation** using Keras' `ImageDataGenerator`. This introduces variability in the training data through random transformations (e.g., rotations, shifts, flips, zooms).

### 2. Deep Feature Extraction
- We utilize pre-trained, state-of-the-art convolutional neural networks (CNNs) as **feature extractors**. Models such as **MobileNet** and **Xception** (trained on ImageNet) are evaluated for this purpose.
- The final classification layers are removed, and a **Global Average Pooling (GAP)** layer is appended to the base model. This step reduces the spatial dimensions of the extracted feature maps, producing a compact, 1D feature vector for each input image.
- These high-level, deep features are extracted for both the training and validation datasets, forming our initial feature space.

### 3. Evolutionary Feature Selection
To reduce dimensionality and eliminate redundant features, a custom Genetic Algorithm is implemented for feature selection.

- **Population Initialization:** A population of individuals is generated, where each individual is a binary-encoded vector. Each bit represents the presence (`1`) or absence (`0`) of a specific feature in the subset.
- **Fitness Evaluation:** The fitness of each individual (feature subset) is calculated by training an **SVM classifier** exclusively on the selected features. The primary fitness metric is the classifier's accuracy on the validation set.
- **Parent Selection:** A tournament selection strategy is employed. For each new offspring, two random individuals are chosen from the population, and the one with the higher fitness score is selected as a parent.
- **Crossover:** A **single-point crossover** operation is performed on pairs of selected parents. A random crossover point is chosen, and genetic material is swapped between parents to produce two new offspring.
- **Mutation:** Random bit flips are applied to the offspring with a low probability, introducing genetic diversity into the population and preventing premature convergence to a local optimum.
- **Iteration:** This process of selection, crossover, and mutation repeats for a set number of generations, evolving the population towards an optimal feature subset.

### 4. Final Classification
The feature subset identified by the genetic algorithm as optimal is used to train a final **SVM classifier**. This model is then evaluated on the test set to report final performance metrics (e.g., accuracy, precision, recall, F1-score).

## Key Features

- **Hybrid Architecture:** Combines the representational power of deep CNNs with the computational efficiency and strong theoretical foundations of SVMs.
- **Optimized Feature Space:** The genetic algorithm performs intelligent feature selection, leading to a model that is potentially more accurate, less complex, and more interpretable.
- **Robustness:** Data augmentation and careful validation ensure the model generalizes well to unseen data.

Key libraries include:
- `tensorflow` / `keras`
- `scikit-learn`
- `numpy`
- `opencv-python`
- `matplotlib`

## Usage

1.  **Prepare Data:** Organize your image dataset into `train` and `validation` directories, with subdirectories for each class.
2.  **Extract Features:** Run the feature extraction script (`feature_extractor.py`) using your chosen pre-trained model (e.g., MobileNet). This will generate `.npy` files containing the features and labels.
3.  **Run Genetic Algorithm:** Execute the genetic algorithm script (`genetic_algorithm.py`) to find the optimal feature subset. This will output a binary mask of the selected features.
4.  **Train Final Model:** Use the `classifier.py` script to train and evaluate the final SVM model on the selected features.

## Results

The model achieves competitive accuracy on the garbage classification task. The evolutionary feature selection step consistently leads to a significant reduction in the number of features used (often >50%) while maintaining or improving classification performance compared to using the full feature set.

## Future Work

- Experiment with other pre-trained models (e.g., EfficientNet, ResNet).
- Incorporate multi-objective optimization in the GA to simultaneously maximize accuracy and minimize feature count.
- Develop an end-to-end deep learning model that integrates the feature selection process directly into the network architecture.


