
# House Price Prediction Using Machine Learning

In this project, our focus is on predicting house prices using machine learning. We have explored different models such as Linear Regression, Decision Trees, and Random Forests to determine which one performs the best. The dataset we are using is sourced from the UCI Machine Learning Repository.


## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Models Used](#models-used)
- [Installation](#installation)
- [Model Comparison](#model-comparison)
- [Conclusion](#conclusion)
- [Future Work](#future-work)
## Project Overview
House Price Prediction is a regression problem where the objective is to predict the price of a house based on different features. This project explains the end-to-end method of building a machine learning model, including data preprocessing, model training, and evaluation.
## Dataset
The dataset utilized in this project is from the UCI Machine Learning Repository.It includes various features of houses and their corresponding prices. The Dataset sourced from [here](https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data)
 
## Models Used
1. **Linear Regression** : A straightforward approach to modeling the relationship between the target variable and predictors.

2. **Decision Tree** : A model that splits the dataset into different subsets based on feature values, forming a tree-like structure.

3. **Random Forest** : An ensemble of decision trees that helps improve the model's accuracy and reduce overfitting.
## Installation
To run this project,Follow the below steps
1. Clone the repository:
   ```bash
   git clone https://github.com/abdultalha0862/House-Price-Prediction-Using-Machine-Learning.git
   cd House-Price-Prediction-Using-Machine-Learning

   ```

2. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```
## Model Comparison
Here's how the models performed:

| Model              | R²       | MAE       | MSE        | RMSE     |
|--------------------|----------|-----------|------------|----------|
| Linear Regression  | 0.668759 | 3.189092  | 24.291119  | 4.928602 |
| Decision Tree      | 0.857963 | 2.394118  | 10.416078  | 3.227395 |
| Random Forest      | 0.891903 | 2.047412  | 7.927145   | 2.815519 |


## Future Work
- **Hyperparameter Tuning:** Further fine-tune the models to improve accuracy.
- **Additional Features:** Consider adding more features to enhance predictions.
- **Model Deployment:** Work towards deploying the model as a web application for real-time predictions.