# Import necessary libraries
import streamlit as st         # For creating the web app interface
import pickle                  # For loading saved model/data files

# Load the pickled movies DataFrame
movies = pickle.load(open('movies_list.pkl', 'rb'))

# Load the cosine similarity matrix
similarity = pickle.load(open('similarity.pkl', 'rb'))

# Extract all movie titles from the DataFrame to show in the dropdown
movies_list = movies['original_title'].values

# Set the main header of the Streamlit app
st.header("🎬 Welcome to the Movie Recommender System")

# Create a dropdown (selectbox) with all movie titles
selected_value = st.selectbox("Select a movie", movies_list)

# Function to recommend similar movies
def recommend(original_title):
    # Get the index of the selected movie in the DataFrame
    index = movies[movies['original_title'] == original_title].index[0]
    
    # Use cosine similarity scores for that movie and sort them in descending order
    distance = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
    
    # Prepare a list to store recommended movie titles
    recommend_movies = []

    # Loop through top 5 most similar movies (skip index 0 since it's the same movie)
    for i in distance[1:6]:
        # Append the movie title to the list using .iloc for row access by index
        recommend_movies.append(movies.iloc[i[0]].original_title)
    
    # Return the list of recommended movies
    return recommend_movies

# When the user clicks the "Recommend" button
if st.button("Recommend"):
    # Call the recommend function with the selected movie
    movie_recommend = recommend(selected_value)

    # Create 5 columns for showing 5 recommended movie titles side by side
    col1, col2, col3, col4, col5 = st.columns(5)

    # Display each recommended movie title in its respective column
    with col1:
        st.text(movie_recommend[0])
    with col2:
        st.text(movie_recommend[1])
    with col3:
        st.text(movie_recommend[2])
    with col4:
        st.text(movie_recommend[3])
    with col5:
        st.text(movie_recommend[4])
