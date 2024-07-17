import pickle
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from PIL import Image

# Charger les fichiers pickle
movies_dict = pickle.load(open("movies.pkl", 'rb'))
similarity_data = pickle.load(open("similarity.pkl", 'rb'))

# Extraire les données de similarité
X = similarity_data['X']
movie_mapper = similarity_data['movie_mapper']
movie_inv_mapper = similarity_data['movie_inv_mapper']

# Convertir le dictionnaire des films en DataFrame
movies = pd.DataFrame(movies_dict)

# Fonction pour recommander des films
def recommend(movie_title, k=5):
    movie_id = movies[movies['title'] == movie_title]['movieId'].values[0]
    neighbour_ids = find_similar_movies(movie_id, X, movie_mapper, movie_inv_mapper, k)
    recommended_movies = [movies[movies['movieId'] == id]['title'].values[0] for id in neighbour_ids]
    return recommended_movies

# Fonction pour trouver les films similaires
def find_similar_movies(movie_id, X, movie_mapper, movie_inv_mapper, k, metric='cosine'):
    X = X.T
    neighbour_ids = []
    movie_ind = movie_mapper[movie_id]
    movie_vec = X[movie_ind]

    if isinstance(movie_vec, (np.ndarray)):
        movie_vec = movie_vec.reshape(1, -1)

    kNN = NearestNeighbors(n_neighbors=k+1, algorithm="brute", metric=metric)
    kNN.fit(X)
    neighbour = kNN.kneighbors(movie_vec, return_distance=False)

    for i in range(0, k):
        n = neighbour.item(i)
        neighbour_ids.append(movie_inv_mapper[n])
    neighbour_ids.pop(0)

    return neighbour_ids

# Configuration de la page Streamlit
st.set_page_config(page_title="Recommandeur de Films", page_icon=":clapper:", layout="wide")

# Titre et description
col1, col2 = st.columns([1, 3])
with col1:
    image = Image.open("data/movie_icon.jpg")
    st.image(image, use_column_width=True)

with col2:
    st.title("Système de Recommandation de Films")
    st.write("Obtenez des recommandations de films personnalisées en fonction de vos préférences.")

# Sélection du film
selected_movie_name = st.selectbox("Sélectionnez un Film", movies['title'].values)

# Bouton pour générer des recommandations
if st.button("Recommander"):
    recommendations = recommend(selected_movie_name)
    st.subheader(f"Voici quelques recommandations basées sur '{selected_movie_name}':")
    for i, movie in enumerate(recommendations, start=1):
        st.success(f"{i}. {movie}")

# Footer
st.markdown("""
<style>
footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    background-color: #f8f9fa;
    color: #6c757d;
    text-align: center;
    padding: 10px;
}
</style>
<footer>
    <p>&copy; DGI❤️ESP </p>
</footer>
""", unsafe_allow_html=True)
