"""
feature_engineering.py

Functions for transforming Netflix data
into machine learning features.
"""


import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder



def create_text_feature(df):
    """
    Combine important columns
    for content-based recommendation.
    """

    df = df.copy()

    df["text"] = (
        df["listed_in"].fillna("") 
        + " "
        + df["description"].fillna("")
    )

    return df



def encode_target(df):
    """
    Encode Movie and TV Show labels.
    """

    encoder = LabelEncoder()

    df["target"] = encoder.fit_transform(
        df["type"]
    )

    return df, encoder



def create_tfidf_features(
        df,
        max_features=5000):

    """
    Convert text data into TF-IDF vectors.
    """

    tfidf = TfidfVectorizer(
        stop_words="english",
        max_features=max_features
    )


    tfidf_matrix = tfidf.fit_transform(
        df["text"]
    )


    return tfidf_matrix, tfidf



def create_user_item_matrix(
        ratings):

    """
    Create user-item matrix
    for collaborative filtering.
    """

    user_item = ratings.pivot_table(
        index="user_id",
        columns="movie_id",
        values="rating"
    ).fillna(0)


    return user_item