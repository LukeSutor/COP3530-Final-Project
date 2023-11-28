import numpy as np
from tqdm import tqdm
import sys
import os
# below line is to allow imports from parent directory
sys.path[0]+= '/../'
from data.dataset import Dataset


class Matrix_Factorization():

    def __init__(self, name, K=100, steps=500, alpha=0.001, beta=0.001, save_interval=25):
        self.name = name
        # Check if the model has already been trained, load if so
        if os.path.exists(f"matrix_factorization/models/{name}/P.csv"):
            print("Model already trained, loading from csv")
            self.load_from_csv()
            return
        else:
            # make new model directory
            os.mkdir(f"matrix_factorization/models/{name}")
            print("Model not found, making new model")
        self.dataset = Dataset()
        self.K = K
        self.steps = steps
        self.alpha = alpha
        self.beta = beta
        self.save_interval = save_interval
        self.R = self.dataset.get_train_matrix()
        self.num_users, self.num_movies = self.R.shape
        self.P = np.random.rand(self.num_users, K)
        self.Q = np.random.rand(self.num_movies, K)
        self.bu = np.zeros(self.num_users)
        self.bi = np.zeros(self.num_movies)
        self.mu = np.mean(self.R[np.where(self.R != 0)])

    def fit(self):
        '''
        Perform matrix factorization to predict empty entries in a matrix.
        '''
        # check if the model has already been trained
        if os.path.exists(f"matrix_factorization/models/{self.name}/P.csv"):
            print("Model already trained")
            return
        
        # create a list of training samples
        samples = [
            (i, j, self.R[i,j])
            for i in range(self.num_users)
            for j in range(self.num_movies)
            if self.R[i,j] > 0
        ]


        # training loop
        progress_bar = tqdm(total=self.steps)
        for i in range(self.steps):
            mse = self.perform_epoch(samples, i)
            progress_bar.set_description(f"MSE: {mse:.4f}")
            progress_bar.update(1)

            # save the current state of the model every save_interval steps
            if i % self.save_interval == 0:
                self.save_as_csv()

        progress_bar.close()

        # save the final state of the model
        self.save_as_csv()
        print("Training complete, model saved under matrix_factorization/model/" + self.name)
    
    def perform_epoch(self, samples, step):
        '''
        Perform one epoch of training
        '''
        np.random.shuffle(samples)
        self.sgd(samples, self.alpha, self.beta)
        mse = self.mse(self.R)
        return mse
    
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
        np.savetxt(f"matrix_factorization/models/{self.name}/P.csv", self.P, delimiter=",")
        np.savetxt(f"matrix_factorization/models/{self.name}/Q.csv", self.Q, delimiter=",")
        np.savetxt(f"matrix_factorization/models/{self.name}/bu.csv", self.bu, delimiter=",")
        np.savetxt(f"matrix_factorization/models/{self.name}/bi.csv", self.bi, delimiter=",")
        np.savetxt(f"matrix_factorization/models/{self.name}/mu.csv", [self.mu], delimiter=",")

    def load_from_csv(self):
        '''
        Load all necessary data from csv files
        '''
        self.P = np.loadtxt(f"matrix_factorization/models/{self.name}/P.csv", delimiter=",")
        self.Q = np.loadtxt(f"matrix_factorization/models/{self.name}/Q.csv", delimiter=",")
        self.bu = np.loadtxt(f"matrix_factorization/models/{self.name}/bu.csv", delimiter=",")
        self.bi = np.loadtxt(f"matrix_factorization/models/{self.name}/bi.csv", delimiter=",")
        self.mu = np.loadtxt(f"matrix_factorization/models/{self.name}/mu.csv", delimiter=",")

def train():
    mf = Matrix_Factorization("Train_1", save_interval=1)
    mf.fit()
    # mf.print_results()
    print(mf.full_matrix())

if __name__ == "__main__":
    train()