# Machine Learning Capstone Project

## Overview
This project is focused on building a machine learning model for credit risk prediction. The project involves extensive data analysis, feature engineering, and model deployment. It provides a structured approach to processing large datasets, training multiple models, and deploying the best performing one.

## Features
- Exploratory Data Analysis (EDA)
- Feature Engineering & Selection
- Model Training and Evaluation (XGBoost, LightGBM, Random Forest)
- Automated Model Deployment Pipeline
- Deployed API for real-time predictions

## Repository Structure
- `notebooks/`: Jupyter notebooks for EDA, feature engineering, and modeling
- `utilities_eda.py`: Utility functions for data preprocessing and analysis
- `data/`: Placeholder for datasets
- `models/`: Saved models and related scripts
- `model_deployment/test_api.py`: Script to test the deployed API

## Installation
1. Clone the repository:
   ```
   bash
   git clone https://github.com/pboric/Project-11x.git
   
2. Install the required packages:
   ```
   bash
   pip install -r requirements.txt
   

## Usage
1. Navigate through the notebooks to understand the data analysis and modeling process.
2. Use the `utilities_eda.py` for data preprocessing.
3. Modify and run the scripts to train and evaluate models on your dataset.

## Deployed API
The trained model is deployed and can be accessed at:
[Home Credit Group Defaulter Prediction API](https://home-credit-group-defaulter-prediction.onrender.com)

To test the API, use the `test_api.py` script located in the `model_deployment` folder:
   ```
   bash
   python model_deployment/test_api.py
   ```

## License
This project is licensed under the MIT License.

## Author
Petar Krešimir Borić
