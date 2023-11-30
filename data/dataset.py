import pandas as pd

class Dataset():

    def __init__(self):
        self.train_dataframe = pd.read_csv('data/train.csv')
        self.test_dataframe = pd.read_csv('data/test.csv')
        self.train_matrix = self.train_dataframe.pivot_table(index='userId', columns='movieId', values='rating').fillna(0)
        self.test_matrix = self.test_dataframe.pivot_table(index='userId', columns='movieId', values='rating').fillna(0)


    def get_train_dataframe(self):
        '''
        Get the training data as a pandas dataframe
        '''
        return self.train_dataframe
    
    def get_test_dataframe(self):
        '''
        Get the test data as a pandas dataframe
        '''
        return self.test_dataframe
    
    def get_train_matrix(self):
        '''
        Get the training data
        '''
        return self.train_matrix.to_numpy()
    
    def get_test_matrix(self):
        '''
        Get the test data
        '''
        return self.test_matrix.to_numpy()
    
    def get_random_n_users(self, n):
        '''
        Get a random sample of n users from the testing data and return as array

        Array properties:
        0th index: userId
        1st index: movieId
        2nd index: rating
        '''
        return self.test_dataframe.sample(n).to_numpy()
    
    def get_movie_name(self, id):
        '''
        Get the movie name based on id, return empty string if not found
        '''
        movies_dataframe = pd.read_csv('data/movies_filtered.csv')
        movie = movies_dataframe.loc[movies_dataframe['movieId'] == id]
        if movie.empty:
            return ''
        else:
            return movie['title'].iloc[0]
        
    def get_user_id(self, user_id):
        '''
        Given a user's id, find their row index in the matrix
        '''
        # loop through rows in the train matrix, return the index if the user id matches
        index = 0
        for i, _ in self.train_matrix.iterrows():
            if i == user_id:
                return index
            index += 1
        return -1
    
    def get_movie_id(self, movie_id):
        '''
        Given a movie's id, find their column index in the matrix
        '''
        # loop through columns in the train matrix, return the index if the movie id matches
        index = 0
        for i in self.train_matrix.columns:
            if i == movie_id:
                return index
            index += 1
        return -1

if __name__ == '__main__':
    dataset = Dataset()
    # print(dataset.get_train_dataframe())
    # print(dataset.get_test_dataframe())
    print(dataset.get_test_matrix().shape)
    # print(dataset.get_movie_id(176371))