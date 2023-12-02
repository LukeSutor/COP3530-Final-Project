import sys
sys.path.insert(0, "C:/Users/Luke/Desktop/UF/COP3530/Final-Project/k_nearest_neighbors")
from data.dataset import Dataset
#from knn import K_Nearest_Neighbors
from matrix_factorization.matrix_factorization import Matrix_Factorization
from k_nearest_neighbors.knn import K_Nearest_Neighbors
from user_interface.user_interface import User_Interface

def main():
    # initialize dataset, matrix factorization, knn, and user interface
    dataset = Dataset()
    mf = Matrix_Factorization("Train_3")
    knn = K_Nearest_Neighbors()
    ui = User_Interface()

    # set the matrix factorization, knn. and dataset in the user interface
    ui.set_dataset(dataset)
    ui.set_mf(mf)
    ui.set_knn(knn)


    # create the window
    ui.create_window()



if __name__ == "__main__":
    main()