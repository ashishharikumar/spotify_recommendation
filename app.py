
import streamlit as st
import pandas as pd
import joblib

from sklearn.neighbors import NearestNeighbors

# ---------------- PAGE CONFIGURATION ----------------

st.set_page_config(
    page_title="Spotify Music Recommender",
    page_icon="🎵",
    layout="wide"
)

# ---------------- CUSTOM DESIGN ----------------

st.markdown("""
<style>

.stApp {
    background-color: #101010;
    color: white;
}

[data-testid="stSidebar"] {
    background-color: #181818;
}

.main-title {
    font-size: 42px;
    font-weight: 800;
    color: #1DB954;
}

.subtitle {
    color: #b3b3b3;
    font-size: 17px;
}

.song-card {
    background-color: #202020;
    padding: 20px;
    border-radius: 14px;
    margin-bottom: 12px;
    border: 1px solid #333333;
}

.song-name {
    font-size: 19px;
    font-weight: bold;
    color: white;
}

.artist-name {
    color: #b3b3b3;
    font-size: 14px;
}

.section-title {
    color: #1DB954;
    font-size: 25px;
    font-weight: bold;
}

div.stButton > button {
    background-color: #1DB954;
    color: black;
    border-radius: 25px;
    border: none;
    font-weight: bold;
    padding: 10px 25px;
}

div.stButton > button:hover {
    background-color: #1ed760;
    color: black;
}

</style>
""", unsafe_allow_html=True)

# ---------------- LOAD DATA ----------------

artifacts = joblib.load("spotify_artifacts.pkl")

df = artifacts["df"].reset_index(drop=True)
X_scaled = artifacts["X_scaled"]

audio_features = [
    "danceability",
    "energy",
    "key",
    "loudness",
    "mode",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "duration_ms"
]

# ---------------- SIDEBAR ----------------

st.sidebar.title("🎧 Spotify")
st.sidebar.markdown("---")

st.sidebar.subheader("Explore Music")

genres = ["All Genres"] + sorted(
    df["playlist_genre"].dropna().unique().tolist()
)

selected_genre = st.sidebar.selectbox(
    "Choose a Genre",
    genres
)

st.sidebar.markdown("---")

st.sidebar.caption("Spotify Songs Recommendation System")
st.sidebar.caption("Discover music that matches your taste.")

# ---------------- MAIN HEADER ----------------

st.markdown(
    '<div class="main-title">🎵 Discover Your Next Favourite Song</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Find songs similar to the music you already love.</div>',
    unsafe_allow_html=True
)

st.write("")

# ---------------- DATASET INFORMATION ----------------

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Songs", f"{len(df):,}")

with col2:
    st.metric(
        "Music Genres",
        df["playlist_genre"].nunique()
    )

with col3:
    st.metric(
        "Playlist Subgenres",
        df["playlist_subgenre"].nunique()
    )

st.markdown("---")

# ---------------- SONG SELECTION ----------------

st.markdown(
    '<div class="section-title">Choose Your Song</div>',
    unsafe_allow_html=True
)

if selected_genre == "All Genres":
    filtered_df = df
else:
    filtered_df = df[
        df["playlist_genre"] == selected_genre
    ]

if filtered_df.empty:
    st.warning("No songs found in this genre.")
    st.stop()

selected_index = st.selectbox(
    "Select a song",
    filtered_df.index.tolist(),
    format_func=lambda i:
        str(df.iloc[i]["track_name"]) +
        " — " +
        str(df.iloc[i]["track_artist"])
)

selected_song = df.iloc[selected_index]

# ---------------- SELECTED SONG DETAILS ----------------

st.write("")

st.markdown(
    '<div class="section-title">Now Selected</div>',
    unsafe_allow_html=True
)

st.markdown(
    f"""
    <div class="song-card">
        <div class="song-name">
            🎵 {selected_song['track_name']}
        </div>
        <div class="artist-name">
            Artist: {selected_song['track_artist']}
        </div>
        <br>
        <b>Genre:</b> {selected_song['playlist_genre']}
        &nbsp;&nbsp;&nbsp;
        <b>Subgenre:</b> {selected_song['playlist_subgenre']}
    </div>
    """,
    unsafe_allow_html=True
)

# ---------------- RECOMMENDATION ----------------

if st.button("✨ Recommend Similar Songs"):

    selected_cluster = selected_song["Cluster"]

    cluster_indices = df.index[
        df["Cluster"] == selected_cluster
    ].tolist()

    cluster_features = X_scaled[cluster_indices]

    model = NearestNeighbors(
        n_neighbors=min(6, len(cluster_indices)),
        metric="cosine"
    )

    model.fit(cluster_features)

    local_index = cluster_indices.index(selected_index)

    distances, indices = model.kneighbors(
        cluster_features[local_index].reshape(1, -1)
    )

    recommendations = []

    for position in indices[0]:

        original_index = cluster_indices[position]

        if original_index != selected_index:
            recommendations.append(
                df.iloc[original_index]
            )

        if len(recommendations) == 5:
            break

    st.markdown("---")

    st.markdown(
        '<div class="section-title">✨ Recommended For You</div>',
        unsafe_allow_html=True
    )

    if recommendations:

        for start in range(0, len(recommendations), 2):

            cols = st.columns(2)

            for col, song in zip(
                cols,
                recommendations[start:start + 2]
            ):

                with col:
                    st.markdown(
                        f"""
                        <div class="song-card">
                            <div class="song-name">
                                🎵 {song['track_name']}
                            </div>
                            <div class="artist-name">
                                {song['track_artist']}
                            </div>
                            <br>
                            <b>Genre:</b> {song['playlist_genre']}<br>
                            <b>Subgenre:</b> {song['playlist_subgenre']}<br>
                            <b>Popularity:</b> {song['track_popularity']}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

        # ---------------- AUDIO FEATURE TABLE ----------------

        st.markdown("---")

        st.markdown(
            '<div class="section-title">🎼 Audio Feature Comparison</div>',
            unsafe_allow_html=True
        )

        feature_data = []

        for song in recommendations:
            row = {
                "Song": song["track_name"],
                "Artist": song["track_artist"]
            }

            for feature in audio_features:
                row[feature] = song[feature]

            feature_data.append(row)

        feature_df = pd.DataFrame(feature_data)

        st.dataframe(
            feature_df,
            use_container_width=True,
            hide_index=True
        )

    else:
        st.info("No recommendations available.")
