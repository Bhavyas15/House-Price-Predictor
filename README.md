## Overview

The House Price Predictor is a machine learning project designed to predict house prices based on various features such as location, square footage, number of bedrooms, and more. This project includes a web application built using Streamlit, allowing users to interact with the model and get price predictions in real-time.

## Features

- **Data Preprocessing**: The dataset is cleaned and preprocessed to handle missing values, outliers, and categorical variables.
- **Machine Learning Model**: A regression model is trained to predict house prices based on the input features.
- **Streamlit Web App**: A user-friendly web interface where users can input house details and get instant price predictions.
- **Deployment**: The model is deployed using Streamlit and can be accessed online.

## Project Structure
House-Price-Predictor/
│
├── data/ # Contains the dataset used for training and testing
│ └── housing_data.csv # Sample dataset
│
├── models/ # Contains the trained model
│ └── model.pkl # Serialized model file
│
├── notebooks/ # Jupyter notebooks for data exploration and model training
│ └── House_Price_Prediction.ipynb
│
├── app.py # Streamlit application script
│
├── requirements.txt # List of dependencies
│
└── README.md # Project documentation

## Getting Started

### Prerequisites

- Python 3.7 or higher
- pip (Python package installer)

### Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/Bhavyas15/House-Price-Predictor.git
   cd House-Price-Predictor

2. **Install Dependencies**
   ```bash
    pip install -r requirements.txt
3. **Run the Streamlit app**
   ```bash
   streamlit run app.py

### Usage
Input House Details: Enter the required details such as location, square footage, number of bedrooms, etc.

Get Prediction: Click on the "Predict" button to get the estimated house price.

### Deployment
The web app is deployed using Streamlit and can be accessed online at House Price Predictor.
https://houses-price-predictor.streamlit.app/

### Acknowledgments
Dataset sourced from Kaggle. (https://www.kaggle.com/c/house-prices-advanced-regression-techniques) 

Streamlit for the web app framework.

Scikit-learn for the machine learning model.
