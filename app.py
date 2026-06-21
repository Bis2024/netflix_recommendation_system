# app.py

import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ===============================
# Page Setup
# ===============================

st.set_page_config(
    page_title="Netflix Recommendation System",
    layout="wide"
)


# ===============================
# Netflix Style CSS
# ===============================

st.markdown("""
<style>
.stApp {
    background-color: #0b0b0b;
    color: white;
}

.main-title {
    font-size: 52px;
    font-weight: 900;
    color: #e50914;
    text-align: center;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    color: #cccccc;
    font-size: 18px;
    margin-bottom: 35px;
}

.movie-card {
    background-color: #181818;
    padding: 20px;
    border-radius: 12px;
    margin-bottom: 18px;
    border: 1px solid #333333;
}

.movie-title {
    color: #ffffff;
    font-size: 24px;
    font-weight: 700;
}

.genre {
    color: #e50914;
    font-weight: 600;
}

.description {
    color: #cccccc;
    font-size: 15px;
}

.stButton > button {
    background-color: #e50914;
    color: white;
    border-radius: 8px;
    border: none;
    padding: 10px 25px;
    font-weight: bold;
}

.stButton > button:hover {
    background-color: #b20710;
    color: white;
}
</style>
""", unsafe_allow_html=True)


# ===============================
# Load Data
# ===============================

@st.cache_data
def load_data():
    df = pd.read_csv("./data/netflix_titles.csv")

    df["director"] = df["director"].fillna("")
    df["cast"] = df["cast"].fillna("")
    df["country"] = df["country"].fillna("")
    df["listed_in"] = df["listed_in"].fillna("")
    df["description"] = df["description"].fillna("")

    # Better content feature for recommendation
    df["content"] = (
        df["listed_in"] + " " +
        df["description"] + " " +
        df["director"] + " " +
        df["cast"] + " " +
        df["country"]
    )

    return df


df = load_data()


# ===============================
# Build Recommendation Model
# ===============================

@st.cache_resource
def build_model(data):
    tfidf = TfidfVectorizer(
        stop_words="english",
        max_features=10000
    )

    tfidf_matrix = tfidf.fit_transform(data["content"])

    cosine_sim = cosine_similarity(
        tfidf_matrix,
        tfidf_matrix
    )

    indices = pd.Series(
        data.index,
        index=data["title"]
    ).drop_duplicates()

    return cosine_sim, indices


cosine_sim, indices = build_model(df)


# ===============================
# Recommendation Function
# ===============================

def recommend_for_new_user(user_ratings, n=10):
    weighted_scores = []
    total_weight = 0

    for title, rating in user_ratings.items():

        if title in indices:
            movie_idx = indices[title]

            # Convert rating into preference weight
            # 1 star = strong dislike
            # 2 star = dislike
            # 3 star = neutral
            # 4 star = like
            # 5 star = strong like
            weight = rating - 3

            # Ignore neutral ratings
            if weight != 0:
                weighted_scores.append(
                    cosine_sim[movie_idx] * weight
                )

                total_weight += abs(weight)

    if len(weighted_scores) == 0:
        return pd.DataFrame()

    final_scores = sum(weighted_scores) / total_weight

    scores = pd.Series(
        final_scores,
        index=df.index
    )

    # Remove movies already selected by user
    selected_indices = []

    for title in user_ratings.keys():
        if title in indices:
            selected_indices.append(indices[title])

    scores = scores.drop(
        labels=selected_indices,
        errors="ignore"
    )

    top_indices = scores.sort_values(
        ascending=False
    ).head(n).index

    return df.loc[top_indices, [
        "title",
        "type",
        "listed_in",
        "description"
    ]]


# ===============================
# Session State
# ===============================

if "users" not in st.session_state:
    st.session_state.users = {}

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "current_user" not in st.session_state:
    st.session_state.current_user = ""


# ===============================
# Header
# ===============================

st.markdown(
    "<div class='main-title'>NETFLIX RECOMMENDER</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='subtitle'>Sign up, rate a few movies, and discover what to watch next.</div>",
    unsafe_allow_html=True
)


# ===============================
# Login / Sign Up Page
# ===============================

if not st.session_state.logged_in:

    col1, col2, col3 = st.columns([1, 1.2, 1])

    with col2:
        menu = st.radio(
            "Choose Option",
            ["Login", "Sign Up"],
            horizontal=True
        )

        if menu == "Sign Up":
            st.subheader("Create New Account")

            new_username = st.text_input("Create Username")
            new_password = st.text_input(
                "Create Password",
                type="password"
            )

            if st.button("Sign Up"):
                if new_username == "" or new_password == "":
                    st.warning("Please enter username and password.")

                elif new_username in st.session_state.users:
                    st.error("Username already exists. Please login.")

                else:
                    st.session_state.users[new_username] = new_password
                    st.success("Account created successfully. Please login now.")

        else:
            st.subheader("Login to Continue")

            username = st.text_input("Username")
            password = st.text_input(
                "Password",
                type="password"
            )

            if st.button("Login"):
                if (
                    username in st.session_state.users
                    and st.session_state.users[username] == password
                ):
                    st.session_state.logged_in = True
                    st.session_state.current_user = username
                    st.rerun()
                else:
                    st.error("Invalid username or password.")

    st.stop()


# ===============================
# Main App After Login
# ===============================

st.sidebar.success(
    f"Logged in as {st.session_state.current_user}"
)

if st.sidebar.button("Logout"):
    st.session_state.logged_in = False
    st.session_state.current_user = ""
    st.rerun()


st.subheader("Choose 3 Movies or Shows You Like")


# Clean movie names for display but keep original names for recommendation
movie_map = {
    title.replace("#", ""): title
    for title in sorted(df["title"].unique())
}

movie_titles = list(movie_map.keys())


col1, col2, col3 = st.columns(3)

with col1:
    movie1 = st.selectbox(
        "Movie / Show 1",
        movie_titles,
        index=0
    )

    rating1 = st.slider(
        f"Rate {movie1}",
        1,
        5,
        5
    )

with col2:
    movie2 = st.selectbox(
        "Movie / Show 2",
        movie_titles,
        index=1
    )

    rating2 = st.slider(
        f"Rate {movie2}",
        1,
        5,
        4
    )

with col3:
    movie3 = st.selectbox(
        "Movie / Show 3",
        movie_titles,
        index=2
    )

    rating3 = st.slider(
        f"Rate {movie3}",
        1,
        5,
        3
    )


# ===============================
# Recommendation Button
# ===============================

if st.button("Recommend Movies"):

    new_user = {
        movie_map[movie1]: rating1,
        movie_map[movie2]: rating2,
        movie_map[movie3]: rating3
    }

    recommendations = recommend_for_new_user(
        new_user,
        n=10
    )

    st.subheader("Recommended For You")

    if recommendations.empty:
        st.warning(
            "Please give at least one movie a rating above or below 3."
        )

    else:
        for _, row in recommendations.iterrows():

            clean_title = row["title"].replace("#", "")

            st.markdown(f"""
            <div class="movie-card">
                <div class="movie-title">{clean_title}</div>
                <p><b>Type:</b> {row['type']}</p>
                <p class="genre">{row['listed_in']}</p>
                <p class="description">{row['description']}</p>
            </div>
            """, unsafe_allow_html=True)