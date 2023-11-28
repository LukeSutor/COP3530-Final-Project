import pandas as pd

class Dataset():

    def __init__(self):
        self.train_dataframe = pd.read_csv('data/train.csv')
        self.test_dataframe = pd.read_csv('data/test.csv')

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
        Get the training data as a numpy matrix
        '''
        train_matrix = self.train_dataframe.pivot_table(index='userId', columns='movieId', values='rating').fillna(0)
        train_matrix.to_csv('data/train_matrix.csv')
        train_matrix = train_matrix.to_numpy()
        # get rid of the first column (userIds)
        train_matrix = train_matrix[:,1:]
        return train_matrix
    
    def get_random_n_users(self, n):
        '''
        Get a random sample of n users from the testing data and return as python array
        '''
        return self.test_dataframe.sample(n).to_numpy()
    

if __name__ == '__main__':
    dataset = Dataset()
    # print(dataset.get_train_dataframe())
    # print(dataset.get_test_dataframe())
    print(dataset.get_train_matrix())
    # print(dataset.get_random_n_users(10))