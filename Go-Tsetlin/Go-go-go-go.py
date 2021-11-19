import string
from time import time
import numpy as np

from sklearn.model_selection import train_test_split

from pyTsetlinMachine.tm import MultiClassConvolutionalTsetlinMachine2D
from process_go_ds import create_TM_representations


game_size = 9  # determines number of columns and rows in the game, this must match with the dataset we are loading
# sgf_dir = '../go9-large'  # here you speciy your Go dataset sgf files directory
sgf_dir = '/Users/alex/Desktop/go9_all'  # here you speciy your Go dataset sgf files directory

''' 
Use "win", "black", or "white" to pick prediction model.
'''
X_linear, Y_temp = create_TM_representations(sgf_dir, game_size, "black")  # where X is all samples represented in bits, Y is all the labels

# reshape from linear to matrix representation
X_temp = X_linear[:, :].reshape(X_linear.shape[0], 9, 18)

# split into test / train and shuffle
X_train, X_test, Y_train, Y_test = train_test_split(X_temp, Y_temp, test_size=0.33, random_state=42, shuffle=True)

N_clauses = 200
Threshold = 200
Forget_rate = 50
ctm = MultiClassConvolutionalTsetlinMachine2D(N_clauses, Threshold, Forget_rate, (9, 18), boost_true_positive_feedback=0)

f = open("log.txt", "a")
f.write(f"Number_of_clauses = {N_clauses}, "+ f"T = {Threshold}, "+ f"S = {Forget_rate}" + "\n")
f.close()


results = np.zeros(0)
print("Training...")
epochs = 100
for i in range(100):
	start = time()
	ctm.fit(X_train, Y_train, epochs=epochs)
	stop = time()

	results = np.append(results, 100*(ctm.predict(X_test) == Y_test).mean())
	print("#%d Mean Accuracy (%%): %.2f; Std.dev.: %.2f; Training Time: %.1f ms/epoch" % (i+1, np.mean(results), np.std(results), (stop-start)/epochs))

	f = open("log.txt", "a")
	f.write("#%d Mean Accuracy (%%): %.2f; Std.dev.: %.2f; Training Time: %.1f ms/epoch" % (i + 1, np.mean(results), np.std(results), (stop - start)) + "\n")
	f.close()