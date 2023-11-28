from data import Dataset
from matrix_factorization import Matrix_Factorization
from k_nearest_neighbors import K_Nearest_Neighbors
from user_interface import User_Interface

def main():
    # initialize dataset, matrix factorization, knn, and user interface
    dataset = Dataset()
    mf = Matrix_Factorization()
    knn = K_Nearest_Neighbors()
    ui = User_Interface()



if __name__ == "__main__":
    main()