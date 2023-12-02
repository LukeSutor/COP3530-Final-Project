import pandas as pd
import numpy as np
from tqdm import tqdm
import sys
import os
sys.path[0]+= '/../'
from data.dataset import Dataset
from data.filter_data import get_matrix

class K_Nearest_Neighbors():

    def __init__(self):
        self.average = 0.00
        self.dataset = Dataset()
        self.m = self.dataset.get_train_matrix()
        self.top = [] 
        self.d = {}

    def get_predicted_score(self, movie_col):
        # use the 5 closest avgs to see what those users rated and return the avg

        pass



    def KNN(self, user_row, movie_col, k=5):
        
        # i is the index of row, going through the differnet users
        for user_index in range(self.m.shape[0]):

            total_user_diff = 0.0
            counter = 0

            # j is the index of the col, going through the different users
            for movie_index in range(self.m.shape[1]):

                if movie_index != movie_col:
                    # user_row doesn't change only the movie index does
                    perm_user_score = self.m[user_row][movie_index]

                    i = self.m[user_index][movie_index]

                    # the user we are on has scored the certain movie (0 means has not scored)
                    if i != 0:

                        real_diff_in_scores = abs(perm_user_score - i)
                        total_user_diff += real_diff_in_scores
                        counter += 1

                    # we have gone through all movies by a certain user
                    if counter != 0:
                        avg = total_user_diff / counter
                        
                        # there isn't even 5 neighbors yet
                        if len(self.top) < 5:

                            self.top.append(avg)
                            self.top.sort()
                            # store what user is associated with this avg
                            self.d[avg] = user_index
                        
                        # the greatest neighbor needs to be replaced
                        elif self.top[4] > avg:
                            self.top[4] = avg
                            self.top.sort()
                            self.d[avg] = user_index
        
        total = 0.0
        c = 0

        for avg in self.top:

            user_index = self.d[avg]
            movie_score = self.m[user_index][movie_col]
            
            if movie_score != 0:
                total += movie_score
                c += 1

        if c == 0:
            return -1
        
        return float(total / c)
    
    

            




                
            

              

    
    

                