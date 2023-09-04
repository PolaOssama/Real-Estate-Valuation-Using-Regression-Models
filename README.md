# House Cost Prediction: Real Estate Valuation

A machine learning repository focused on predicting real estate valuations based on various features provided by the UCI Machine Learning Repository.

## Overview

This project aims to predict real estate valuation based on features like transaction date, house age, distance to the nearest MRT station, number of convenience stores, latitude, and longitude. Both manual modeling and Orange data mining software were used in the analysis. The dataset underwent preprocessing to ensure optimal model performance. Various regression models were applied, with Random Forest yielding the best results.

## Table of Contents

- [Dataset Details](#dataset-details)
- [Preprocessing Steps](#preprocessing-steps)
- [Models Used](#models-used)
- [Analysis with Orange](#analysis-with-orange)
- [Getting Started](#getting-started)
- [Prerequisites](#prerequisites)
- [Results](#results)
- [Contributing](#contributing)
- [Acknowledgments](#acknowledgments)
- [License](#license)

## Dataset Details

The dataset was sourced from the [UCI Machine Learning Repository - Real estate valuation data set](https://archive.ics.uci.edu/ml/datasets/Real+estate+valuation+data+set#). It contains the following features:

1. Transaction Date: Exact date of the transaction.
2. House Age: Age of the house (in years).
3. Distance to Nearest MRT Station: Distance to the nearest MRT station.
4. Number of Convenience Stores: Number of convenience stores in the living circle on foot.
5. Latitude: Geographical coordinate.
6. Longitude: Geographical coordinate.

The target variable is:
House Price of Unit Area: House price per unit area.

## Preprocessing Steps

During the preprocessing phase, the following steps were taken:

1. Data Cleaning: Removal of any null or missing values.
2. Feature Engineering: Conversion of the transaction date to more usable forms and creation of new features.
3. Feature Scaling: Standardization and normalization of numerical attributes.

## Models Used

1. XGBoost: An optimized gradient boosting library.
2. Lasso Regression: Linear Model trained with L1 prior as regularizer.
3. Linear Regression: Simple linear approach for predictions.
4. Decision Tree: Non-parametric supervised learning approach.
5. Ridge Regression: Linear least squares with l2 regularization.
6. Random Forest: Ensemble learning method with multiple decision trees.

## Analysis with Orange

Orange, a visual programming software for data science, was utilized for additional data exploration, preprocessing, and model evaluation:

1. Data Exploration: Gaining insights into data distributions, patterns, and correlations.
2. Data Preprocessing: Handling missing values and feature scaling.
3. Model Evaluation: Comparing the performance of various models.

## Getting Started

1. Clone the repo:
   https://github.com/PolaOssama/Real-Estate-Valuation-Using-Regression-Models
2. Install the required packages

## Prerequisites

Ensure you have the following software/libraries installed:

- Python
- Orange
- XGBoost
- Scikit-learn

## Results

After applying various regression models to the dataset, the **Random Forest** emerged as the best-performing model for predicting real estate valuations.

### Model Performance:

**Random Forest:**
- R-squared (R²) Value: 79.5%

This indicates that approximately 79.5% of the variance in the dependent variable can be explained by the features in our model using the Random Forest regression.

## Contributing

Contributions are what make the open-source community an inspiring place. Any contributions you make are greatly appreciated.

1. Fork the Project
2. Create your Feature Branch
3. Commit your Changes
4. Push to the Branch
5. Open a Pull Request

## Acknowledgments

- UCI Machine Learning Repository for providing the dataset.
- Orange for their intuitive data analysis software.

## License

Distributed under the MIT License. See `LICENSE` for more information.
