import pandas as pd

def filter():
    '''
    Load raw ratings and movie data, filter out movies with less than 5000 
    ratings and users with less than 600 ratings, and save filtered data to csv
    '''
    # load in ratings.cvs data
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


def make_matrix():
    '''
    Make a pivot table of the ratings data
    '''
    # load in ratings.cvs data
    ratings = pd.read_csv('data/ratings_filtered.csv')

    # create a pivot table with userId as the rows, movieId as the columns, and ratings as the values
    ratings_matrix = ratings.pivot_table(index='userId', columns='movieId', values='rating')
    # fill in all NaN values with 0
    ratings_matrix.fillna(0, inplace=True)

    # save matrix to csv
    ratings_matrix.to_csv('data/ratings_matrix.csv')


def get_numpy():
    '''
    Get a numpy array of the ratings matrix
    '''
    # load in ratings_matrix.cvs data
    ratings_matrix = pd.read_csv('data/ratings_matrix.csv')
    # get numpy array
    ratings_matrix = ratings_matrix.to_numpy()
    # get rid of the first column (userIds)
    ratings_matrix = ratings_matrix[:,1:]
    return ratings_matrix



if __name__ == '__main__':
    # filter()
    # make_matrix()
    print(get_numpy())