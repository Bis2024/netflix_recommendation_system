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
    page_icon="🎬",
    layout="wide"
)


# ===============================
# Netflix Style CSS
# ===============================

st.markdown("""
<style>
/* Main app */
.stApp {
    background:
        radial-gradient(circle at 20% 10%, rgba(229, 9, 20, 0.20), transparent 25%),
        radial-gradient(circle at 80% 0%, rgba(120, 0, 0, 0.18), transparent 25%),
        linear-gradient(180deg, #050505 0%, #0b0b0b 45%, #111111 100%);
    color: white;
}

/* Hide Streamlit default padding a little */
.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* Hero section */
.hero {
    padding: 55px 30px 45px 30px;
    border-radius: 24px;
    background:
        linear-gradient(rgba(0,0,0,0.50), rgba(0,0,0,0.75)),
        linear-gradient(135deg, rgba(229,9,20,0.30), rgba(20,20,20,0.95));
    border: 1px solid rgba(255,255,255,0.10);
    box-shadow: 0 25px 80px rgba(0,0,0,0.50);
    margin-bottom: 35px;
}

.logo {
    color: #e50914;
    font-size: 18px;
    font-weight: 900;
    letter-spacing: 4px;
    margin-bottom: 10px;
}

.main-title {
    color: #ffffff;
    font-size: 58px;
    line-height: 1.05;
    font-weight: 900;
    margin-bottom: 12px;
}

.highlight {
    color: #e50914;
}

.subtitle {
    color: #d0d0d0;
    font-size: 18px;
    max-width: 760px;
    line-height: 1.6;
    margin-bottom: 22px;
}

.chip {
    display: inline-block;
    background: rgba(255,255,255,0.10);
    border: 1px solid rgba(255,255,255,0.12);
    color: #eeeeee;
    padding: 8px 13px;
    border-radius: 999px;
    margin-right: 8px;
    margin-bottom: 8px;
    font-size: 13px;
}

/* Auth box */
.auth-card {
    background: rgba(24,24,24,0.92);
    padding: 30px;
    border-radius: 18px;
    border: 1px solid rgba(255,255,255,0.12);
    box-shadow: 0 18px 60px rgba(0,0,0,0.45);
}

/* Section headings */
.section-title {
    font-size: 28px;
    font-weight: 800;
    color: #ffffff;
    margin-top: 20px;
    margin-bottom: 8px;
}

.section-subtitle {
    color: #bfbfbf;
    margin-bottom: 22px;
}

/* Input panel */
.input-panel {
    background: rgba(24,24,24,0.88);
    padding: 22px;
    border-radius: 18px;
    border: 1px solid rgba(255,255,255,0.10);
    min-height: 170px;
    box-shadow: 0 14px 45px rgba(0,0,0,0.30);
}

.panel-number {
    color: #e50914;
    font-weight: 900;
    font-size: 13px;
    letter-spacing: 2px;
}

/* Recommendation cards */
.movie-card {
    background:
        linear-gradient(180deg, rgba(35,35,35,0.98), rgba(18,18,18,0.98));
    padding: 22px;
    border-radius: 18px;
    margin-bottom: 18px;
    border: 1px solid rgba(255,255,255,0.12);
    box-shadow: 0 16px 45px rgba(0,0,0,0.35);
    min-height: 250px;
    transition: transform 0.2s ease, border 0.2s ease;
}

.movie-card:hover {
    transform: translateY(-4px);
    border: 1px solid rgba(229,9,20,0.65);
}

.movie-title {
    color: #ffffff;
    font-size: 22px;
    font-weight: 850;
    margin-bottom: 9px;
}

.badge {
    display: inline-block;
    background: #e50914;
    color: white;
    padding: 4px 9px;
    border-radius: 7px;
    font-size: 12px;
    font-weight: 800;
    margin-bottom: 12px;
}

.genre {
    color: #ff4b55;
    font-weight: 700;
    font-size: 13px;
    margin-top: 8px;
    margin-bottom: 10px;
}

.description {
    color: #cfcfcf;
    font-size: 14px;
    line-height: 1.55;
}

/* Streamlit widgets */
.stButton > button {
    background: linear-gradient(90deg, #e50914, #b20710);
    color: white;
    border-radius: 10px;
    border: none;
    padding: 12px 28px;
    font-weight: 800;
    box-shadow: 0 10px 25px rgba(229,9,20,0.25);
}

.stButton > button:hover {
    background: linear-gradient(90deg, #ff1722, #e50914);
    color: white;
    border: none;
}

div[data-baseweb="select"] > div {
    background-color: #f4f4f4;
    border-radius: 10px;
}

.stSlider label {
    color: #cfcfcf !important;
}

hr {
    border: none;
    border-top: 1px solid rgba(255,255,255,0.12);
    margin: 25px 0;
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
    df["type"] = df["type"].fillna("Unknown")

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

            # 1 star = dislike, 3 star = neutral, 5 star = strong like
            weight = rating - 3

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

st.markdown("""
<div class="hero">
    <div class="logo">NETFLIX STYLE AI PROJECT</div>
    <div class="main-title">Find your next <span class="highlight">favorite</span> movie.</div>
    <div class="subtitle">
        Sign up, rate three titles, and get personalized recommendations using
        TF-IDF, cosine similarity, and rating-weighted content-based filtering.
    </div>
    <span class="chip">Machine Learning</span>
    <span class="chip">NLP</span>
    <span class="chip">Recommendation System</span>
    <span class="chip">Streamlit App</span>
</div>
""", unsafe_allow_html=True)


# ===============================
# Login / Sign Up Page
# ===============================

if not st.session_state.logged_in:

    col1, col2, col3 = st.columns([1, 1.15, 1])

    with col2:
        st.markdown("<div class='auth-card'>", unsafe_allow_html=True)

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

        st.markdown("</div>", unsafe_allow_html=True)

    st.stop()


# ===============================
# Main App After Login
# ===============================

st.sidebar.success(
    f"Logged in as {st.session_state.current_user}"
)

st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.write(
    "This app recommends Netflix titles based on your selected movies and ratings."
)

if st.sidebar.button("Logout"):
    st.session_state.logged_in = False
    st.session_state.current_user = ""
    st.rerun()


st.markdown("<div class='section-title'>Choose 3 movies or shows you like</div>", unsafe_allow_html=True)
st.markdown("<div class='section-subtitle'>Your ratings now affect the recommendations. A 5-star title has more influence than a 1-star title.</div>", unsafe_allow_html=True)


# Clean movie names for display but keep original names for recommendation
movie_map = {
    title.replace("#", ""): title
    for title in sorted(df["title"].unique())
}

movie_titles = list(movie_map.keys())


col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("<div class='input-panel'><div class='panel-number'>TITLE 01</div>", unsafe_allow_html=True)

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

    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='input-panel'><div class='panel-number'>TITLE 02</div>", unsafe_allow_html=True)

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

    st.markdown("</div>", unsafe_allow_html=True)

with col3:
    st.markdown("<div class='input-panel'><div class='panel-number'>TITLE 03</div>", unsafe_allow_html=True)

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

    st.markdown("</div>", unsafe_allow_html=True)


st.markdown("<br>", unsafe_allow_html=True)


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
        n=12
    )

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Recommended for you</div>", unsafe_allow_html=True)

    if recommendations.empty:
        st.warning(
            "Please give at least one movie a rating above or below 3."
        )

    else:
        rec_cols = st.columns(3)

        for i, (_, row) in enumerate(recommendations.iterrows()):

            clean_title = row["title"].replace("#", "")
            short_description = row["description"]

            if len(short_description) > 210:
                short_description = short_description[:210] + "..."

            with rec_cols[i % 3]:
                st.markdown(f"""
                <div class="movie-card">
                    <div class="movie-title">{clean_title}</div>
                    <div class="badge">{row['type']}</div>
                    <div class="genre">{row['listed_in']}</div>
                    <div class="description">{short_description}</div>
                </div>
                """, unsafe_allow_html=True)

else:
    st.markdown("""
    <div class="movie-card">
        <div class="movie-title">Ready when you are 🍿</div>
        <div class="description">
            Select three titles, rate them, and click <b>Recommend Movies</b>.
            The system will compare your taste with Netflix content and return personalized suggestions.
        </div>
    </div>
    """, unsafe_allow_html=True)
