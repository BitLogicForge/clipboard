# Table of Contents

## Machine Learning Fundamentals

### Existing Lessons

- **Introduction to Scikit-Learn** (Beginner) - _Suggested, move to existing if created_
  - Covers the basics of using the Scikit-Learn library for common machine learning tasks.
  - **Key Concepts:** Data loading, preprocessing (scaling, encoding), model training (Linear Regression, Logistic Regression, K-Nearest Neighbors), evaluation metrics.
- **Model Evaluation and Selection** (Beginner) - _Suggested, move to existing if created_
  - Focuses on methods for evaluating model performance and selecting the best model.
  - **Key Concepts:** Train/validation/test split, cross-validation, confusion matrix, precision, recall, F1-score, ROC curve, AUC.
- **Hyperparameter Tuning (Grid Search, Random Search)** (Intermediate) - _Suggested, move to existing if created_
  - Techniques for finding the optimal hyperparameters for a machine learning model.
  - **Key Concepts:** Grid Search CV, Randomized Search CV, optimizing model performance.
- **Working with Imbalanced Datasets** (Intermediate) - _Suggested, move to existing if created_
  - Strategies for handling datasets where one class significantly outnumbers others.
  - **Key Concepts:** Undersampling, oversampling (SMOTE), cost-sensitive learning, appropriate evaluation metrics (Precision-Recall curve).

## Supervised Learning

### Regression

#### Existing Lessons

- **Height vs Weight Dataset with Linear Regression**
  - Explores the linear relationship between two continuous variables using a synthetic dataset.
  - **Key Points:** Data generation, fitting a simple linear regression model (y = mx + b), calculating R-squared, visualizing the regression line and data points.
  - Notebook: `height_weight_2d.ipynb`
- **Height, Weight, and Age Dataset with Multiple Regression**
  - Analyzes the relationship between a dependent variable (e.g., weight) and multiple independent variables (height, age).
  - **Key Points:** Extending linear regression to multiple predictors, interpreting coefficients, 3D visualization of data and regression plane.
  - Notebook: `height_weight_age_3d.ipynb`
- **Height vs Weight Dataset with Polynomial Regression**
  - Models non-linear relationships between variables using polynomial features.
  - **Key Points:** Generating polynomial features, fitting models of varying degrees, comparing model fit (e.g., using R-squared), visualizing the polynomial curve, understanding overfitting.
  - Notebook: `height_weight_polynomial.ipynb`
- **Exponential Regression with TensorFlow**
  - Demonstrates how to model exponential growth or decay using TensorFlow.
  - **Key Points:** Implementing custom non-linear models, using appropriate loss functions, applying learning rate scheduling for better convergence, visualizing the fitted exponential curve and loss history.
  - Notebook: `tensor2u2.ipynb`

### Classification

#### Existing Lessons

- **Height-Weight BMI Category Classification**
  - Classifies individuals into BMI categories based on height and weight using machine learning.
  - **Key Points:** Feature engineering (calculating BMI), data preprocessing, training a Random Forest classifier, evaluating performance using accuracy and confusion matrix.
  - Notebook: `height_weight_bmi_classification.ipynb`
- **Introduction to TensorFlow 2 and Classification**
  - Introduces the fundamentals of TensorFlow 2 for building and training neural networks for classification.
  - **Key Points:** TensorFlow basics (tensors, variables), building a sequential model with Dense layers, using activation functions (ReLU, Softmax), compiling with loss (Categorical Crossentropy) and optimizer (Adam), training on the MNIST handwritten digit dataset, evaluating accuracy.
  - Notebook: `tensor2.ipynb`
- **Breast Cancer Classification with TensorFlow**
  - Implements a binary classification model using TensorFlow to predict breast cancer malignancy.
  - **Key Points:** Using the Wisconsin Breast Cancer dataset (from scikit-learn), data scaling/normalization, building a binary classifier, training the model, evaluating with metrics like precision, recall, F1-score.
  - Notebook: `tensor2u.ipynb`

#### Suggested Additional Lessons

- **Decision Trees and Random Forests** (Beginner)
  - Explores tree-based models for classification and regression.
  - **Key Concepts:** Decision tree structure, splitting criteria (Gini impurity, entropy), pruning, ensemble methods, Random Forest algorithm, feature importance.
- **Support Vector Machines (SVM)** (Intermediate)
  - Introduces SVMs for classification and regression tasks.
  - **Key Concepts:** Maximal margin classifier, support vectors, kernel trick (linear, polynomial, RBF), applications in high-dimensional spaces.
- **Bayesian Methods in Machine Learning** (Advanced)
  - Probabilistic approach to machine learning.
  - **Key Concepts:** Bayes' theorem, prior/posterior distributions, Bayesian inference, Naive Bayes classifiers, Gaussian Processes.

## Unsupervised Learning

### Existing Lessons

- **Analyzing Word Similarity with Pre-trained GloVe Embeddings** (_Also relevant to NLP_)
  - Demonstrates using pre-trained GloVe embeddings for semantic analysis.
  - **Key Points:** Loading embeddings, calculating cosine similarity, finding analogous words (e.g., king - man + woman = queen), visualizing word clusters using t-SNE.
  - Notebook: `analyze_glove_embeddings.ipynb`

### Suggested Additional Lessons

- **Clustering Algorithms (K-Means, DBSCAN)** (Beginner)
  - Covers unsupervised learning techniques for grouping data points.
  - **Key Concepts:** K-Means algorithm, choosing K (elbow method), DBSCAN algorithm, density-based clustering, noise points.
- **Dimensionality Reduction (PCA, t-SNE)** (Intermediate)
  - Techniques for reducing the number of features while preserving important information.
  - **Key Concepts:** Principal Component Analysis (PCA), variance explained, t-Distributed Stochastic Neighbor Embedding (t-SNE) for visualization.
- **Anomaly Detection** (Intermediate)
  - Techniques for identifying rare items, events, or observations which raise suspicions.
  - **Key Concepts:** Statistical methods, Isolation Forest, One-Class SVM.
- **Autoencoders for Feature Learning** (Advanced)
  - Unsupervised neural networks used for learning efficient data codings.
  - **Key Concepts:** Encoder, decoder, bottleneck layer, reconstruction loss, applications in dimensionality reduction and denoising.

## Deep Learning

### Core Concepts & Frameworks

#### Existing Lessons

- **Introduction to TensorFlow 2 and Classification** (_Listed under Supervised Learning as well_)
  - Introduces the fundamentals of TensorFlow 2 for building and training neural networks for classification.
  - **Key Points:** TensorFlow basics (tensors, variables), building a sequential model with Dense layers, using activation functions (ReLU, Softmax), compiling with loss (Categorical Crossentropy) and optimizer (Adam), training on the MNIST handwritten digit dataset, evaluating accuracy.
  - Notebook: `tensor2.ipynb`

#### Suggested Additional Lessons

- **Introduction to PyTorch** (Intermediate)
  - Covers the fundamentals of PyTorch for building deep learning models.
  - **Key Concepts:** Tensors, autograd, building neural networks (nn.Module), optimizers, loss functions.

### Architectures

#### Existing Lessons

- **Time Series Forecasting with LSTMs** (_Also relevant to Sequential Data_)
  - Covers the application of Long Short-Term Memory (LSTM) networks for predicting future values in sequential data.
  - **Key Points:** Preparing time series data (windowing, scaling), building an LSTM model architecture, training on sequential data, evaluating forecasting accuracy (e.g., MAE, RMSE), visualizing predictions vs actual values.
  - Notebook: `time_series_forecasting_lstm.ipynb`

#### Suggested Additional Lessons

- **Introduction to Convolutional Neural Networks (CNNs)** (Intermediate)
  - Learn the architecture and application of CNNs, primarily for image recognition tasks.
  - **Key Concepts:** Convolutional layers, pooling layers, feature extraction, image classification (e.g., using CIFAR-10 dataset).
- **Recurrent Neural Networks (RNNs)** (Intermediate)
  - Introduces RNNs for processing sequential data, preceding LSTMs.
  - **Key Concepts:** Sequence modeling, vanishing/exploding gradients problem, basic RNN cell structure.
- **Generative Adversarial Networks (GANs)** (Advanced)
  - Learn how GANs work and how to build them for generating novel data, particularly images.
  - **Key Concepts:** Generator network, discriminator network, adversarial training process, applications in image synthesis and style transfer.
- **Graph Neural Networks (GNNs)** (Advanced)
  - Applying deep learning techniques to graph-structured data.
  - **Key Concepts:** Node embeddings, graph convolutions, applications in social networks, molecular structures.
- **Object Detection with YOLO/SSD** (Advanced)
  - Building models that can detect and locate objects within images.
  - **Key Concepts:** Bounding boxes, anchor boxes, popular architectures like YOLO (You Only Look Once) and SSD (Single Shot MultiBox Detector).

## Natural Language Processing (NLP)

### Existing Lessons

- **Analyzing Word Similarity with Pre-trained GloVe Embeddings** (_Listed under Unsupervised Learning as well_)
  - Demonstrates using pre-trained GloVe embeddings for semantic analysis.
  - **Key Points:** Loading embeddings, calculating cosine similarity, finding analogous words (e.g., king - man + woman = queen), visualizing word clusters using t-SNE.
  - Notebook: `analyze_glove_embeddings.ipynb`

### Suggested Additional Lessons

- **Sentiment Analysis with NLTK/spaCy** (Beginner)
  - Basic NLP techniques for determining the sentiment expressed in text.
  - **Key Concepts:** Tokenization, stop word removal, stemming/lemmatization, bag-of-words, TF-IDF, using libraries like NLTK or spaCy.
- **Natural Language Processing with Transformers** (Advanced)
  - Explore state-of-the-art transformer models for various NLP tasks.
  - **Key Concepts:** Attention mechanism, pre-trained models (BERT, GPT), fine-tuning for tasks like text classification, sentiment analysis, or question answering.

## Specialized Topics & Applications

### Suggested Additional Lessons

- **Reinforcement Learning Basics** (Intermediate)
  - Introduces the fundamental concepts of reinforcement learning where agents learn through interaction with an environment.
  - **Key Concepts:** Agents, environments, states, actions, rewards, Q-learning, policy gradients, using libraries like OpenAI Gym.
- **Recommender Systems (Collaborative Filtering, Content-Based)** (Intermediate)
  - Building systems that predict user preferences.
  - **Key Concepts:** User-item interactions, collaborative filtering (user-based, item-based), content-based filtering, matrix factorization.
- **Explainable AI (XAI) with SHAP/LIME** (Advanced)
  - Techniques for understanding and interpreting the predictions of complex machine learning models.
  - **Key Concepts:** Model interpretability, SHAP (SHapley Additive exPlanations), LIME (Local Interpretable Model-agnostic Explanations).
- **Federated Learning** (Advanced)
  - Training machine learning models across multiple decentralized devices holding local data samples, without exchanging them.
  - **Key Concepts:** Data privacy, distributed training, model aggregation.

## Deployment

### Suggested Additional Lessons

- **Deploying Machine Learning Models (Flask/FastAPI)** (Intermediate)
  - Making trained models available for use via APIs.
  - **Key Concepts:** Saving/loading models, creating web APIs using Flask or FastAPI, handling requests and responses.

---

_Note: Some lessons fit into multiple categories and have been noted._
_Suggested lessons can be moved to the relevant "Existing Lessons" section once the corresponding notebook is created._
