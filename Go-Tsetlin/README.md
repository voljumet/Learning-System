# Go Tsetlin prediction
The main file of this project is Go-go-go-go.py

Some options will have to be chosen in the terminal when running the file.
* What move to predict? (Odd numbers to predict Balck stone, even numbers to predict Whire stone. If 0 is chosen, it will use ALL moves recursively)
* What to predict?: (win / black (next move)/ white (next move)) 
* Load data from folder (needs processing) = F, from preprocessed datafile = D: 
This file will need to be configured before running, the following hyperparameters are used: 
    
     # to set the size of the board
    game_size = 9

     # to set the directory of the data files
    sgf_dir = 'go9'

     # to set the number of clauses
    clauses = 200 #

     # to set the threshold
    Threshold = 300

     # to set the forget rate
    Forget_rate = 10

    # to set the epochs
    epochs = 250

---
