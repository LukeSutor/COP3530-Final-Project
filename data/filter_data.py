import pandas as pd

def filter():
    '''
    Load raw ratings and movie data, filter out movies with less than 5000 
    ratings and users with less than 600 ratings, and save filtered data to csv
    '''
    # load in ratings.csv data
    ratings = pd.read_csv('data/ratings.csv')

    # filter out movies with less than 5000 ratings
    ratings = ratings.groupby('movieId').filter(lambda x: len(x) >= 5000)
    # filter out users with less than 600 ratings
    ratings = ratings.groupby('userId').filter(lambda x: len(x) >= 600)
    # get rid of the timestamp column
    ratings.drop('timestamp', axis=1, inplace=True)

    # print stats
    users = len(ratings['userId'].unique())
    movies = len(ratings['movieId'].unique())
    print("Users: ", users)
    print("Movies: ", movies)
    print(len(ratings), " ratings and ", users*movies, " possible ratings")

    # save filtered data to csv
    ratings.to_csv('data/ratings_filtered.csv', index=False)

    
    # load in movies.csv data
    movies = pd.read_csv('data/movies.csv')
    # filter out movies with less than 5000 ratings
    movies = movies[movies['movieId'].isin(ratings['movieId'])]
    # get rid of the genres column
    movies.drop('genres', axis=1, inplace=True)
    # save filtered data to csv
    movies.to_csv('data/movies_filtered.csv', index=False)


def make_train_test():
    '''
    Make train and test sets from the filtered ratings data
    '''
    # load in ratings.csv data
    ratings = pd.read_csv('data/ratings_filtered.csv')

    # split into train and test sets
    train = ratings.sample(frac=0.99, random_state=200)
    test = ratings.drop(train.index)

    # save train and test sets to csv
    train.to_csv('data/train.csv', index=False)
    test.to_csv('data/test.csv', index=False)


def get_matrix(train=True):
    '''
    Make pivot tables, convert to numpy, and return them
    '''
    # load proper data
    if train:
        data = pd.read_csv('data/train.csv')
    else:
        data = pd.read_csv('data/test.csv')
    # make pivot table
    train_matrix = data.pivot_table(index='userId', columns='movieId', values='rating')
    train_matrix.to_csv('data/train_matrix.csv')
    train_matrix = train_matrix.to_numpy()
    # get rid of the first column (userIds)
    train_matrix = train_matrix[:,1:]
    return train_matrix



if __name__ == '__main__':
    # filter()
    # make_train_test()
    pass