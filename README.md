# Board-Games-Recommender-System
This repository contains a comprehensive project for a basic board game recommendation system using machine learning clustering techniques. The system leverages the [BoardGameGeek database](https://www.kaggle.com/datasets/seanthemalloy/board-game-geek-database) to recommend similar games based on user selection.

---

## Table of Contents

1. [Overview](#overview)
2. [Project Files](#project-files)
3. [Requirements](#requirements)
4. [Setup Instructions](#setup-instructions)
5. [Usage](#usage)
6. [Conclusions](#conclusions)

---

## Overview

This project implements a clustering-based recommender system for board games. Users can select a board game and receive recommendations based on similar game attributes such as ratings, game complexity, playtime, game cathegories and mechanics. The application is built using Python and Streamlit.

---

## Project Files

This repository contains the following files:

- **`app.py`**: The main Streamlit application script to run the recommender system.
- **`requirements.txt`**: A list of Python dependencies required for the project.
- **`ML and Board Games.pdf`**: A presentation detailing the project objectives, implementation, and conclusions.
- **`MLClusteringBG.ipynb`**: The Jupyter Notebook containing detailed steps for data cleaning, exploratory data analysis, data preprocessing, data modeling and clustering analysis for the recommender system.
- **`boardgamesdf.csv`**: A preprocessed dataset containing information about board games, including their names, ratings, weights, min/max number of players, and other features. This dataset is generated as the output of the `MLClusteringBG.ipynb` notebook and serves as the primary data source for the recommender system.
  
---

## Requirements

To run this project, ensure you have the following installed:

- Python 3.8 or above
- pip (Python package installer)

You can find the specific dependencies in the `requirements.txt` file.

---

## Setup Instructions

1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd <repository-directory>

2. **Create and Activate a Virtual Environment**:
   It is recommended to use a virtual environment to manage dependencies and avoid conflicts with other Python projects on your system.

   - Create a virtual environment:
     ```bash
     python -m venv env
     ```
   - Activate the virtual environment:
     - On Windows:
       ```bash
       .\env\Scripts\activate
       ```
     - On macOS/Linux:
       ```bash
       source env/bin/activate
       ```
       
3. **Install Dependencies**:
   Use pip to install the required Python packages:
   ```bash
   pip install -r requirements.txt

4. **Prepare the Dataset**:
   - Ensure the dataset (`boardgamesdf.csv`) is located in the same directory as `app.py`.
   - If the dataset is in a different location, update the path in `app.py` to reflect its location:
     ```python
     df = pd.read_csv("<path-to-your-dataset>/boardgamesdf.csv")
     ```

5. **Run the Application**:
   Launch the Streamlit app using the following command:
   ```bash
   streamlit run app.py

6. **Deactivate the Virtual Environment**:
   If you no longer need the virtual environment, deactivate it by running:
   ```bash
   deactivate
   
---

## Usage

1. **Launch the App**:
   Follow the setup instructions to start the Streamlit application.

2. **Select a Board Game**:
   - Use the dropdown menu in the app to choose a board game.

3. **View Recommendations**:
   - The application displays the top 10 rated games similar to your selection.
   - Recommendations are based on clustering analysis of game features.
     
---

## Conclusions

1. **Feature Selection**:
   - **Numerical Features**: The final dataset for this project included 6 numerical features: `game weight` (representing the complexity of the game, on a scale from 1 to 5), `min players`, `max players`, `playtime`, `age recommendation`, and `number of expansions`. These were log-transformed, robust-scaled, and min-max scaled to ensure consistency and compatibility with the clustering algorithm.
   - **Categorical Features**: The dataset also included 360 categorical features derived from `game categories`, `game subcategories`, `game mechanics`, and `game themes`. These features were already encoded prior to preprocessing.
   - A mixed features matrix was constructed, combining these scaled numerical features and encoded categorical data, ensuring all data was normalized between 0 and 1 for optimal clustering performance.
  
2. **Clustering Effectiveness**:
   - The use of K-means clustering with PCA-reduced data resulted in moderate segmentation of board games into clusters based on features like game weight, playtime, game cathegories and mechanics.
   - Optimal clustering parameters were determined through the silhouette score and elbow method.
   - However, the clustering results revealed large clusters, with up to 8461 games in the first cluster and 2529 in the smallest, making recommendations less precise without additional filtering.

3. **Recommendation Accuracy**:
   - The system recommends similar board games by leveraging cluster membership.
   - To simplify the recommender system, only the top 10 games within a cluster, sorted by the highest average rating, are displayed.
   - The current recommender system could be further improved by integrating similarity measures such as cosine similarity within clusters to refine results.

4. **Scalability**:
   - Designed to handle large datasets, ensuring broad applicability for various user preferences.
  
5. **Future Directions**:
   - Explore cumulative explained variance to determine the optimal number of PCA components.
   - Experiment with alternative clustering models such as Agglomerative Clustering, DBSCAN, or HDBSCAN to improve cluster precision.
   - Refine the recommender system by incorporating additional steps to measure similarity within clusters and improve overall accuracy.

6. **User Experience**:
   - Integrated Streamlit for an intuitive and interactive user interface, making game selection and recommendations seamless.

