import streamlit as st
import pickle
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------
# Load your dictionary pickle
# ---------------------------------------------------------
movie_dict = pickle.load(open('movies_dict.pkl', 'rb'))
movies = pd.DataFrame(movie_dict)
movies['title'] = movies['title'].astype(str).str.strip()

# ---------------------------------------------------------
# Vectorize tags and compute similarity
# ---------------------------------------------------------
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(movies['tags']).toarray()
similarity = cosine_similarity(vectors)

# ---------------------------------------------------------
# Streamlit Page Config
# ---------------------------------------------------------
st.set_page_config(
    page_title="Netflix Movie Recommender",
    layout="wide",
    page_icon="🎬"
)

# ---------------------------------------------------------
# CINEMATIC CSS
# ---------------------------------------------------------
st.markdown("""
<style>

body {
    background: linear-gradient(135deg, #000000, #1a1a1a, #000000);
    background-size: 400% 400%;
    animation: gradientMove 12s ease infinite;
}

@keyframes gradientMove {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}

.main {
    background-color: transparent;
    color: white;
}

/* CINEMATIC HERO */
.hero {
    padding: 70px 20px;
    text-align: center;
    background: radial-gradient(circle, rgba(229,9,20,0.4) 0%, rgba(0,0,0,0.9) 70%);
    border-radius: 20px;
    margin-bottom: 50px;
    box-shadow: 0 0 40px rgba(229,9,20,0.4);
}

.hero-title {
    font-size: 70px;
    font-weight: 900;
    color: #E50914;
    text-shadow: 0 0 25px rgba(229,9,20,0.8);
    letter-spacing: 3px;
}

.hero-sub {
    font-size: 24px;
    color: #f2f2f2;
    margin-top: 15px;
    text-shadow: 0 0 10px rgba(255,255,255,0.4);
}

/* MOVIE STRIP DIVIDER */
.strip {
    height: 8px;
    background: repeating-linear-gradient(
        90deg,
        #E50914 0px,
        #E50914 20px,
        transparent 20px,
        transparent 40px
    );
    margin: 40px 0;
    border-radius: 4px;
}

/* MOVIE CARD */
.movie-card {
    background: rgba(255, 255, 255, 0.07);
    backdrop-filter: blur(12px);
    padding: 22px;
    border-radius: 18px;
    text-align: center;
    transition: 0.35s;
    border: 1px solid rgba(255,255,255,0.15);
    box-shadow: 0 0 25px rgba(0,0,0,0.4);
}

.movie-card:hover {
    transform: scale(1.12);
    box-shadow: 0 0 35px rgba(229,9,20,0.6);
    background: rgba(255, 255, 255, 0.15);
}

.movie-title {
    font-size: 22px;
    font-weight: 800;
    color: #E50914;
    text-shadow: 0 0 10px rgba(229,9,20,0.7);
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------
st.sidebar.title("🎞️ Cinematic Mode")
st.sidebar.info("""
A **Netflix Movie Recommendation System**  
powered by **NLP + Cosine Similarity**.

🎬 Built by Billa Sahithi
🔥 Netflix‑Inspired  
🌌 Cinematic UI  
""")

st.sidebar.write("Made with ❤️ for your ML Portfolio")

# ---------------------------------------------------------
# Hero Section
# ---------------------------------------------------------
st.markdown("""
<div class='hero'>
    <div class='hero-title'>NETFLIX MOVIE RECOMMENDER</div>
    <div class='hero-sub'>Experience movie discovery like a film trailer</div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Recommendation Function
# ---------------------------------------------------------
def recommend(movie):
    if movie not in movies['title'].values:
        return ["Movie not found in database"]

    movie_index = movies[movies['title'] == movie].index[0]
    distances = similarity[movie_index]

    movie_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:6]

    return [movies.iloc[i[0]].title for i in movie_list]

# ---------------------------------------------------------
# UI Components
# ---------------------------------------------------------
selected_movie = st.selectbox(
    "🎥 Choose a Movie",
    movies['title'].values,
    help="Pick a movie to get cinematic recommendations"
)

st.markdown("<div class='strip'></div>", unsafe_allow_html=True)

if st.button("🍿 Show Recommendations"):
    st.markdown("## 🎬 Top 5 Cinematic Recommendations")
    st.write("")

    recommendations = recommend(selected_movie)

    cols = st.columns(5)

    for idx, movie in enumerate(recommendations):
        with cols[idx]:
            st.markdown(f"""
                <div class='movie-card'>
                    <div class='movie-title'>{movie}</div>
                </div>
            """, unsafe_allow_html=True)
