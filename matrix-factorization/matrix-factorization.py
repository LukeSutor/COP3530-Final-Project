import numpy as np
import sys
sys.path[0]+= '/../'
print(sys.path)
from data.filter_data import get_matrix


class Matrix_Factorization():

    def __init__(self, K=100, steps=500, alpha=0.001, beta=0.001, update_interval=10):
        self.K = K
        self.steps = steps
        self.alpha = alpha
        self.beta = beta
        self.update_interval = update_interval
        self.R = get_matrix()
        self.num_users, self.num_movies = self.R.shape
        self.P = np.random.rand(self.num_users, K)
        self.Q = np.random.rand(self.num_movies, K)
        self.bu = np.zeros(self.num_users)
        self.bi = np.zeros(self.num_movies)
        self.mu = np.mean(self.R[np.where(self.R != 0)])

    def fit(self):
        '''
        Perform matrix factorization to predict empty
        entries in a matrix.
        '''
        # create a list of training samples
        samples = [
            (i, j, self.R[i,j])
            for i in range(self.num_users)
            for j in range(self.num_movies)
            if self.R[i,j] > 0
        ]

        # perform stochastic gradient descent for number of steps
        for step in range(self.steps):
            self.perform_epoch(samples, step)
    
    def perform_epoch(self, samples, step):
        '''
        Perform one epoch of training
        '''
        np.random.shuffle(samples)
        self.sgd(samples, self.alpha, self.beta)
        mse = self.mse(self.R)
        if (step+1) % self.update_interval == 0:
            print("Iteration: %d; error = %.4f" % (step+1, mse))
    
    def mse(self, actual):
        '''
        A function to compute the total mean square error
        '''
        # compute mean squared error
        xs, ys = actual.nonzero()
        predicted = self.full_matrix()
        error = 0
        for x, y in zip(xs, ys):
            error += pow(actual[x, y] - predicted[x, y], 2)
        return np.sqrt(error)
    
    def sgd(self, samples, alpha, beta):
        '''
        Perform stochastic gradient descent
        '''
        for i, j, r in samples:
            # compute prediction and error
            prediction = self.get_rating(i, j)
            e = (r - prediction)
            # update biases
            self.bu[i] += alpha * (e - beta * self.bu[i])
            self.bi[j] += alpha * (e - beta * self.bi[j])
            # update user and item latent feature matrices
            self.P[i,:] += alpha * (e * self.Q[j,:] - beta * self.P[i,:])
            self.Q[j,:] += alpha * (e * self.P[i,:] - beta * self.Q[j,:])

    def get_rating(self, i, j):
        '''
        Get the predicted rating of user i and item j
        '''
        prediction = self.mu + self.bu[i] + self.bi[j] + self.P[i,:].dot(self.Q[j,:].T)
        return prediction
    
    def full_matrix(self):
        '''
        Compute the full matrix using the resultant biases, P and Q
        '''
        return self.mu + self.bu[:,np.newaxis] + self.bi[np.newaxis:,] + self.P.dot(self.Q.T)
    
    def print_results(self):
        '''
        Print the results of the matrix factorization
        '''
        print("P x Q:")
        print(self.full_matrix())
        print("Global bias:")
        print(self.mu)
        print("User bias:")
        print(self.bu)
        print("Item bias:")
        print(self.bi)
        print("User latent feature matrix:")
        print(self.P)
        print("Item latent feature matrix:")
        print(self.Q)
        print("Final RMSE:")
        print(self.mse(self.R))

    def save_as_csv(self):
        '''
        Save all necessary data as csv files
        '''
        np.savetxt("data/P.csv", self.P, delimiter=",")
        np.savetxt("data/Q.csv", self.Q, delimiter=",")
        np.savetxt("data/bu.csv", self.bu, delimiter=",")
        np.savetxt("data/bi.csv", self.bi, delimiter=",")
        np.savetxt("data/mu.csv", [self.mu], delimiter=",")

    def load_from_csv(self):
        '''
        Load all necessary data from csv files
        '''
        self.P = np.loadtxt("data/P.csv", delimiter=",")
        self.Q = np.loadtxt("data/Q.csv", delimiter=",")
        self.bu = np.loadtxt("data/bu.csv", delimiter=",")
        self.bi = np.loadtxt("data/bi.csv", delimiter=",")
        self.mu = np.loadtxt("data/mu.csv", delimiter=",")

def train():
    mf = Matrix_Factorization()
    mf.fit()
    mf.print_results()
    mf.save_as_csv()

if __name__ == "__main__":
    train()