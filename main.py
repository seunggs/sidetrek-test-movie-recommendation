import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import difflib
from sklearn.metrics.pairwise import cosine_similarity
import typing
from typing import Tuple, List, Union
import scipy
from dataclasses import dataclass
import pathlib


@dataclass
class Hyperparameters(object):
    """
    datarange: int 
    yearrange: int 
    vectorizer: TfidfVectorizer
    numb: int 
    """
    datarange: int = 10000
    yearrange: int = 1995
    vectorizer: TfidfVectorizer = TfidfVectorizer()
    numb: int = 10

df = pd.read_csv("movies.csv", sep=";")

def process_data(df: pd.DataFrame, datarange: int, yearrange: int) -> Tuple[pd.DataFrame, List[str]]:
    df = df[df["original_language"]=="en"]
    df = df[df["spoken_languages"]=="English"]
    df = df[df["status"]=="Released"]
    df = df[df["production_countries"]=="United States of America"]
    df["release_date"] = pd.to_datetime(df["release_date"])
    df = df[df['release_date'].dt.year>=yearrange]
    df.dropna(subset=['title', 'genres', 'imdb_id'], inplace=True)
    df= df.drop(index=301337)
    df = df.fillna("")
    df["popularity"] = df["popularity"].astype(float)
    df = df.sort_values(by='popularity', ascending=False)
    df = df.drop(["id", "budget", "imdb_id","original_language","original_title","popularity","production_companies","production_countries","release_date","revenue","runtime","spoken_languages","status","vote_average","vote_count","production_companies_number","production_countries_number","spoken_languages_number"], axis=1)
    df = df.iloc[:datarange]
    df = df.reset_index()
    title_df = df["title"]
    title_df_list = title_df.tolist()
    return df, title_df_list

df, title_df_list = process_data(df,10000, 1995)

def vectorize_data(df: pd.DataFrame, vectorizer: TfidfVectorizer) -> scipy.sparse.csr.csr_matrix:
    merged_info = df["genres"] + df["overview"] + df["tagline"]
    return vectorizer.fit_transform(merged_info)

vectorized_data = vectorize_data(df, TfidfVectorizer())


def recommend_movies(df: pd.DataFrame, numb: int, vectorized_data: scipy.sparse.csr.csr_matrix, title_df_list: List[str]) -> List[str]:
    """
    This function keeps asking user to insert a favourite movie, and then recommends `numb` similar movies. 
    When user inserts `quit` the loop breaks and the function prints `Thank you`
    args:
        numb
    returns:
        recommendation
    """
    while True:
        movie_name = input("Insert movie name: ")
        recommendation = []
        if movie_name.lower() == "quit":
            break
        else:
            movie_match = difflib.get_close_matches(movie_name, title_df_list)[0]
            print("Found in Our List:", movie_match,"\n")
            movie_index = df[df["title"]==movie_match].index.values
            movie_index = movie_index[0]
            similarity = cosine_similarity(vectorized_data)
            similarity_score = list(enumerate(similarity[movie_index]))
            sorted_similar_movies = sorted(similarity_score, key = lambda x:x[1], reverse = True)
            sorted_similar_movies = sorted_similar_movies[1:]
            i = 0
            for movie in sorted_similar_movies:
                index = movie[0]
                title_from_index = df[df.index==index]['title'].values[0]
                if (i<numb):
                    print(title_from_index)
                    recommendation.append(title_from_index)
                    i+=1
        print("---------------\n")
    return recommendation

recommend_movies(df, 10, vectorized_data, title_df_list)
