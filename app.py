import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

st.title('Recommender System for Board Games')
st.write('This app provides board games recommendations based on clustering analysis')

df = pd.read_csvdf = pd.read_csv("boardgamesdf.csv")


selected_name = st.selectbox('Select a board game:', df['name'].unique())

if selected_name:
    selected_cluster = df[df['name'] == selected_name]['Cluster'].values[0]
    recommendations = df[(df['Cluster'] == selected_cluster) & (df['name'] != selected_name)]
    top_recommendations = recommendations.sort_values(by='avgrating', ascending=False).head(10)
    st.write(f"Top 10 games similar to '{selected_name}':")
    st.dataframe(top_recommendations[['bggid', 'name', 'avgrating', 'gameweight']])
