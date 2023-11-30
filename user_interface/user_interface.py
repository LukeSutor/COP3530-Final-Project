import tkinter

class User_Interface():

    def __init__(self):
        self.dataset = None
        self.knn = None
        self.mf = None
        self.window = tkinter.Tk()
        self.pairs = None
        self.mf_text = tkinter.Text(self.window, bg="light grey", height=1, width=10, state=tkinter.DISABLED)
        self.actual_text = tkinter.Text(self.window, bg="light grey", height=1, width=10, state=tkinter.DISABLED)
        self.knn_text = tkinter.Text(self.window, bg="light grey", height=1, width=10, state=tkinter.DISABLED)

    def set_dataset(self, dataset):
        self.dataset = dataset

    def set_knn(self, knn):
        self.knn = knn

    def set_mf(self, mf):     
        self.mf = mf
        
    def create_window(self):
        # create the window
        self.window.title("Movie Recommendation Showdown")
        self.window.geometry("800x600")

        # create the widgets
        self.create_layout()

        # run the main loop
        self.window.mainloop()

    def create_layout(self):
        # create the widgets
        self.create_title()
        self.create_refresh()
        self.create_pairs()
        self.create_comparison_fields()

    def create_title(self):
        # create the title
        self.title = tkinter.Label(self.window, text="Welcome to Movie Recommendation Showdown! Click on a user-movie pair to compare the algorithms. Click the refresh button for new pairs.")
        self.title.pack()


    def create_refresh(self):
        # create the refresh button
        self.button2 = tkinter.Button(self.window, text="Refresh", command=self.create_pairs)
        # add a gap underneath
        self.button2.pack(pady=10)

    def create_pairs(self):
        # initialize pairs if not already initialized
        if self.pairs is None:
            self.pairs = self.dataset.get_random_n_users(10)

        # undisplay old pairs
        for widget in self.window.winfo_children():
            try:
                if widget.cget("text").startswith("User ID: "):
                    widget.pack_forget()
            except:
                pass

        # get new pairs
        self.pairs = self.dataset.get_random_n_users(10)

        for i, pair in enumerate(self.pairs):
            user_id = int(pair[0])
            movie_id = int(pair[1])
            movie_name = self.dataset.get_movie_name(movie_id)

            button = tkinter.Button(self.window, text="User ID: " + str(user_id) + " Movie: " + movie_name, command=lambda index=i:  self.run_comparison(index))
            button.pack(pady=5)

    def create_comparison_fields(self):
        # create a text field for knn, matrix factorization, and the actual rating.
        knn_label = tkinter.Label(self.window, text="KNN Prediction: ")
        actual_label = tkinter.Label(self.window, text="Actual Rating: ")
        mf_label = tkinter.Label(self.window, text="MF Prediction: ")

        # pack the text fields in a row horizontally, centered
        knn_label.place(relx=0.25, rely=0.8, anchor=tkinter.CENTER)
        self.knn_text.place(relx=0.25, rely=0.85, anchor=tkinter.CENTER)
        actual_label.place(relx=0.5, rely=0.8, anchor=tkinter.CENTER)
        self.actual_text.place(relx=0.5, rely=0.85, anchor=tkinter.CENTER)
        mf_label.place(relx=0.75, rely=0.8, anchor=tkinter.CENTER)
        self.mf_text.place(relx=0.75, rely=0.85, anchor=tkinter.CENTER)


    
    def run_comparison(self, index):
        # get the button information based on index
        user_id = self.pairs[index][0]
        movie_id = self.pairs[index][1]
        rating = self.pairs[index][2]

        # get the movie and user indices
        user_index = self.dataset.get_user_id(user_id)
        movie_index = self.dataset.get_movie_id(movie_id)

        # get the predictions
        mf_prediction = self.mf.get_rating(user_index-1, movie_index-1)


        # set the text fields
        self.mf_text.config(state=tkinter.NORMAL)
        self.mf_text.delete("1.0", tkinter.END)
        self.mf_text.insert(tkinter.END, mf_prediction)
        self.mf_text.config(state=tkinter.DISABLED)

        self.actual_text.config(state=tkinter.NORMAL)
        self.actual_text.delete("1.0", tkinter.END)
        self.actual_text.insert(tkinter.END, rating)
        self.actual_text.config(state=tkinter.DISABLED)
        


if __name__ == "__main__":
    ui = User_Interface()
    ui.create_window()