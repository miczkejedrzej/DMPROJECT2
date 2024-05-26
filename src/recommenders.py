import pandas as pd
from sklearn.cluster import KMeans
from scipy.sparse import csr_matrix
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split


class RuleBasedRecommender:
    """
    Rule-based recommender class.
    This class is used for rule-based recommendation.
    """


class Data:
    def __init__(self,data_dir:str="ml-latest-small",
                  test_percent:float=0.2,
                  fill_nans_type:str="mean",
                  seed:int=124) -> None:
        """
        Data class for processing and preparing data for clustering-based recommendation.
        
        Parameters:
        - data_dir (str): Directory path where the data files are located.
        - test_percent (float): Percentage of data to be used for testing.
        - fill_nans_type (str): Type of method to fill NaN values in the data table.
        - seed (int): Random seed for reproducibility.
        """
        
        self.fill_nans_type = fill_nans_type
        self.unprocessed_data = pd.merge(pd.read_csv(f'{data_dir}/{"movies"}.csv'),
                             pd.read_csv(f'{data_dir}/{"ratings"}.csv'),
                             on='movieId')
        
        self.unprocessed_data["genres"] = self.unprocessed_data["genres"].str.split("|")
        self.unprocessed_data = self.unprocessed_data.explode("genres")
        self.unprocessed_data = self.unprocessed_data[["movieId", "genres", "userId", "rating"]]

        self.movie_genres = self.unprocessed_data[["movieId", "genres"]].drop_duplicates()
        self.train_data, self.test_data = train_test_split(
                    self.unprocessed_data, test_size=test_percent, random_state=seed)
        
        self.train_data_table_for_clustering = self.train_data.pivot_table(index='userId', 
                                           columns='genres', values='rating',aggfunc="mean")
        
        self.train_data_table_for_clustering = self.fill_nans(self.train_data_table_for_clustering)
        
        
    def fill_nans(self,df) -> pd.DataFrame:
        """
        Fill NaN values in the data table based on the specified method.
        
        Parameters:
        - df (pd.DataFrame): Data table to fill NaN values in.
        
        Returns:
        - pd.DataFrame: Data table with filled NaN values.
        """
        if self.fill_nans_type == "mean":
            f = lambda x: x.fillna(x.mean())
        else:
            f = lambda x: x.fillna(0)
        return df.apply(f,axis=1)


class ClusteringBasedRecommender:
    def __init__(self, data:pd.DataFrame,
                 movie_genres:pd.DataFrame,
                 Clusterer=KMeans,
                 clusterer_params={"n_clusters":10, "random_state":124}) -> None:
        """
        Clustering-based recommender class.
        This class is used for clustering-based recommendation.
        
        Parameters:
        - data (pd.DataFrame): Data table used for clustering.
        - movie_genres (pd.DataFrame): Data table containing movie genres.
        - Clusterer (class): Clustering algorithm class to be used.
        - clusterer_params (dict): Parameters for the clustering algorithm.
        """
        self.data_table = data
        self.movie_genres = movie_genres
        self.clusterer = Clusterer(**clusterer_params)
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
        genre_ratings = self.data_table.loc[users_in_cluster, movie_genres].mean(axis=1).mean()

        return genre_ratings
    