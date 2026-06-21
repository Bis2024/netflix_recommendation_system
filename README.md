## Netflix Recommendation System

A complete end-to-end Netflix Recommendation System project built using Data Analysis, Machine Learning, NLP, Recommendation Systems, Matrix Factorization, and Streamlit.



## Project Overview

This project analyzes Netflix content and builds multiple machine learning models before creating a personalized recommendation engine.

The project includes:

* Data Cleaning
* Exploratory Data Analysis (EDA)
* Feature Engineering
* Text Vectorization (TF-IDF)
* Classification Models
* Hyperparameter Tuning
* Content-Based Recommendation System
* Matrix Factorization (NMF)
* Streamlit Web Application



## Dataset

Dataset Source:

Netflix Titles Dataset from Kaggle

Dataset contains:

* 8,807 Netflix titles
* Movies and TV Shows
* Genres
* Ratings
* Release Years
* Descriptions
* Countries


## Technologies Used

### Python Libraries

* pandas
* numpy
* matplotlib
* seaborn
* scikit-learn
* wordcloud
* streamlit


## Exploratory Data Analysis

The EDA section answers questions such as:

* Movies vs TV Shows distribution
* Top countries producing Netflix content
* Content growth over time
* Rating distribution
* Genre analysis
* Movie duration analysis
* Word cloud visualization

### Key Findings

* Movies represent nearly 70% of Netflix content.
* TV-MA is the most common rating.
* United States contributes the most content.
* Netflix content grew rapidly after 2015.
* International Movies and Dramas dominate the platform.


## Machine Learning Models

### Problem Statement

Predict whether a Netflix title is:

* Movie (0)
* TV Show (1)

### Feature Engineering

Text features were created using:

* Title
* Genre
* Description

TF-IDF Vectorization was applied before training.

### Models Trained

1. Dummy Baseline
2. Logistic Regression
3. Naive Bayes
4. Decision Tree
5. Tuned Decision Tree
6. K-Nearest Neighbors (KNN)

### Results

| Model               | Accuracy |
| ------------------- | -------- |
| Baseline            | 69.64%   |
| Logistic Regression | 99.04%   |
| Naive Bayes         | 96.37%   |
| Decision Tree       | 99.15%   |
| Tuned Decision Tree | 99.60%   |
| KNN                 | 97.21%   |

### Best Model

Tuned Decision Tree achieved the highest accuracy.


## Recommendation System

### Content-Based Filtering

Recommendations are generated using:

* Genres
* Descriptions
* TF-IDF
* Cosine Similarity

Users receive recommendations based on similarity between content.


## Matrix Factorization

To simulate real-world recommendation systems:

* Synthetic user ratings were generated
* User-Item Matrix was created
* Non-Negative Matrix Factorization (NMF) was applied

This allows:

* Hidden preference discovery
* Personalized recommendations
* Predicted ratings for unseen movies


## Streamlit Web App

Features:

* User Sign Up
* User Login
* Movie Selection
* Movie Rating
* Personalized Recommendations
* Netflix-Inspired Interface

---

## Project Structure

Netflix-Recommendation-System/

│

├── app.py

├── netflix_titles.csv

├── netflix_recommendation_system.ipynb

├── requirements.txt

├── README.md

│

└── assets/


## Run Locally

Install requirements:

pip install -r requirements.txt

Run Streamlit app:

streamlit run app.py


## Concepts Demonstrated

* Data Cleaning
* Exploratory Data Analysis
* Feature Engineering
* Natural Language Processing
* TF-IDF
* Classification
* Hyperparameter Tuning
* Recommendation Systems
* Cosine Similarity
* Matrix Factorization
* Streamlit Deployment


## Future Improvements

* Real user database
* Movie posters using TMDB API
* Collaborative Filtering
* Hybrid Recommendation System
* User watch history
* Deployment on Streamlit Cloud