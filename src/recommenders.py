from typing import List
import pandas as pd
from sklearn.cluster import KMeans
from scipy.sparse import csr_matrix
import numpy as np
from sklearn.preprocessing import Normalizer, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer, KNNImputer


class RuleBasedRecommender:
    """
    Rule-based recommender class.
    This class is used for rule-based recommendation.
    """


class Data:
    def __init__(self,data_dir:str="ml-latest-small",
                  test_percent:float=0.2,
                  imputer=SimpleImputer(strategy="mean"),
                  preprocessors:List=[StandardScaler()],
                  seed:int=124) -> None:
        """
        Data class for processing and preparing data for clustering-based recommendation.
        
        Parameters:
        - data_dir (str): Directory path where the data files are located.
        - test_percent (float): Percentage of data to be used for testing.
        - imputer: Imputer object to fill NaN values in the data table.
        - preprocessors: List of preprocessor objects to apply on the data table.
        - seed (int): Random seed for reproducibility.
        """
        
        self.imputer = imputer
        self.preprocessors = preprocessors

        self.unprocessed_data = pd.merge(pd.read_csv(f'{data_dir}/{"movies"}.csv'),
                             pd.read_csv(f'{data_dir}/{"ratings"}.csv'),
                             on='movieId')
        
        self.unprocessed_data = self.unprocessed_data[["movieId", "genres", "userId", "rating"]]
        self.unprocessed_data["genres"] = self.unprocessed_data["genres"].str.split("|")
        self.unprocessed_data = self.unprocessed_data.explode("genres")
        

        self.movie_genres = self.unprocessed_data[["movieId", "genres"]].drop_duplicates()
        self.train_data, self.test_data = train_test_split(
                    self.unprocessed_data, test_size=test_percent, random_state=seed)
        
        self.train_data_table_for_clustering = self.train_data.pivot_table(index='userId', 
                                           columns='genres', values='rating',aggfunc="mean")
        
        #impute NaN values
        imputed_data = imputer.fit_transform(self.train_data_table_for_clustering)
        self.train_data_table_for_clustering = pd.DataFrame(imputed_data, 
                                                            columns=self.train_data_table_for_clustering.columns, 
                                                            index=self.train_data_table_for_clustering.index)

        #apply preprocessors sequentially
        preprocessed_data = self.train_data_table_for_clustering
        for preprocessor in self.preprocessors:
            preprocessed_data = preprocessor.fit_transform(preprocessed_data)

        self.train_data_table_for_clustering_normalized = pd.DataFrame(preprocessed_data,
                                                                      columns=self.train_data_table_for_clustering.columns,
                                                                      index=self.train_data_table_for_clustering.index)


class ClusteringBasedRecommender:
    def __init__(self, data:pd.DataFrame,
                 data_unnormalized:pd.DataFrame,
                 movie_genres:pd.DataFrame,
                 clusterer=KMeans(10, random_state=124)) -> None:
        """
        Clustering-based recommender class.
        This class is used for clustering-based recommendation.
        
        Parameters:
        - data (pd.DataFrame): Data table used for clustering.
        - data_unnormalized (pd.DataFrame): Unnormalized data table used for prediction.
        - movie_genres (pd.DataFrame): Data table containing movie genres.
        - clusterer: Clustering algorithm object to be used.
        """
        self.data_table = data
        self.data_table_unnormalized = data_unnormalized

        self.movie_genres = movie_genres
        self.clusterer = clusterer
        self.clusters = None
        
    def train(self) -> None:
        """
        Train the clustering-based recommender model.
        This method performs clustering on the data table and assigns clusters to each data point.
        """
        self.clusters = self.clusterer.fit_predict(self.data_table)
        self.data_table['cluster'] = self.clusters


    def predict(self, user_id:int, movie_id:int) -> float:
        """
        Predict the rating for a given user and movie.
        
        Parameters:
        - user_id (int): ID of the user.
        - movie_id (int): ID of the movie.
        
        Returns:
        - float: Predicted rating for the user and movie.
        """
        user_cluster = self.clusters[user_id-1]
        movie_genres = self.movie_genres[self.movie_genres['movieId'] == movie_id]['genres'].values

        users_in_cluster = self.data_table[self.data_table['cluster'] == user_cluster].index
        genre_ratings = self.data_table_unnormalized.loc[users_in_cluster, movie_genres].mean(axis=1).mean()
        
        return genre_ratings
    