import string
from datetime import time

import numpy as np

from pyTsetlinMachine.tm import MultiClassConvolutionalTsetlinMachine2D
from process_go_ds import create_TM_representations


game_size = 9  # determines number of columns and rows in the game, this must match with the dataset we are loading
sgf_dir = 'go9'  # here you speciy your Go dataset sgf files directory
X, Y = create_TM_representations(sgf_dir, game_size)  # where X is all samples represented in bits, Y is all the labels


# train_data = np.loadtxt("2DNoisyXORTrainingData.txt")
X_st = X[:, :].reshape(X.shape[0], 18, 9)
Y_st = Y

X_train = X_st[:int(X_st.shape[0] * 0.8)]
Y_train = Y_st[:int(Y_st.shape[0] * 0.8)]

X_test = X_st[int(X_st.shape[0] * 0.8):]
Y_test = Y_st[int(Y_st.shape[0] * 0.8):]

# test_data = np.loadtxt("2DNoisyXORTestData.txt")
# X_test = test_data[:, 0:-1].reshape(test_data.shape[0], 4, 4)
# Y_test = test_data[:, -1]

ctm = MultiClassConvolutionalTsetlinMachine2D(40, 60, 3.9, (2, 2), boost_true_positive_feedback=0)

# ctm.fit(X_train, Y_train, epochs=5000)

results = np.zeros(0)
for i in range(100):
	start = time()
	ctm.fit(X_train, Y_train, epochs=5000)
	stop = time()

	results = np.append(results, 100*(ctm.predict(X_test) == Y_test).mean())
	print("#%d Mean Accuracy (%%): %.2f; Std.dev.: %.2f; Training Time: %.1f ms/epoch" % (i+1, np.mean(results), np.std(results), (stop-start)/5.0))

#
# print("Accuracy:", 100*(ctm.predict(X_test) == Y_test).mean())
#
# Xi = np.array([[[0, 1, 1, 0],
# 		[1, 1, 0, 1],
# 		[1, 0, 1, 1],
# 		[0, 0, 0, 1]]])
#
# print("\nInput Image:\n")
# print(Xi)
# print("\nPrediction: %d" % (ctm.predict(Xi)))
