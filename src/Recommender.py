"""
recommender.py

Contains recommendation functions.
"""

import pandas as pd
import numpy as np


def recommend(title,
              df,
              cosine_sim,
              n=10):

    """
    Recommend similar titles.
    """

    idx = df[df["title"] == title].index[0]

    scores = list(
        enumerate(
            cosine_sim[idx]
        )
    )

    scores = sorted(
        scores,
        key=lambda x: x[1],
        reverse=True
    )

    scores = scores[1:n+1]

    indices = [i[0] for i in scores]

    return df.iloc[indices][
        [
            "title",
            "type",
            "listed_in",
            "description"
        ]
    ]


def recommend_for_new_user(
        ratings,
        df,
        cosine_sim,
        n=10):

    """
    Recommend movies for a new user.
    """

    scores = np.zeros(len(df))

    total = 0

    for movie, rating in ratings.items():

        if movie not in df["title"].values:
            continue

        idx = df[df["title"] == movie].index[0]

        scores += cosine_sim[idx] * rating

        total += rating

    if total > 0:
        scores = scores / total

    recommendations = pd.DataFrame({
        "score": scores
    })

    recommendations["title"] = df["title"]

    recommendations = recommendations.sort_values(
        "score",
        ascending=False
    )

    watched = list(ratings.keys())

    recommendations = recommendations[
        ~recommendations["title"].isin(watched)
    ]

    return df[
        df["title"].isin(
            recommendations.head(n)["title"]
        )
    ]