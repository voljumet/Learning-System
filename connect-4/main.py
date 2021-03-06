from pyTsetlinMachine.tm import MultiClassTsetlinMachine
import numpy as np
from time import time

from keras.datasets import mnist

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

X_train = np.where(X_train.reshape((X_train.shape[0], 28*28)) > 75, 1, 0)
X_test = np.where(X_test.reshape((X_test.shape[0], 28*28)) > 75, 1, 0)

tm = MultiClassTsetlinMachine(2000, 50, 10.0)

print("\nAccuracy over 250 epochs:\n")
for i in range(250):
	start_training = time()
	tm.fit(X_train, Y_train, epochs=1, incremental=True)
	stop_training = time()

	start_testing = time()
	result = 100*(tm.predict(X_test) == Y_test).mean()
	stop_testing = time()

	print("#%d Accuracy: %.2f%% Training: %.2fs Testing: %.2fs" % (i+1, result, stop_training-start_training, stop_testing-start_testing))


# from pyTsetlinMachine.tm import MultiClassTsetlinMachine
# import numpy as np
#
# train_data = np.loadtxt("NoisyXORTrainingData.txt")
# X_train = train_data[:,0:-1]
# Y_train = train_data[:,-1]
#
# test_data = np.loadtxt("NoisyXORTestData.txt")
# X_test = test_data[:,0:-1]
# Y_test = test_data[:,-1]
#
# tm = MultiClassTsetlinMachine(10, 15, 3.9, boost_true_positive_feedback=0)
#
# tm.fit(X_train, Y_train, epochs=200)
#
# print("Accuracy:", 100*(tm.predict(X_test) == Y_test).mean())
#
# print("Prediction: x1 = 1, x2 = 0, ... -> y = %d" % (tm.predict(np.array([[1,0,1,0,1,0,1,1,1,1,0,0]]))))
# print("Prediction: x1 = 0, x2 = 1, ... -> y = %d" % (tm.predict(np.array([[0,1,1,0,1,0,1,1,1,1,0,0]]))))
# print("Prediction: x1 = 0, x2 = 0, ... -> y = %d" % (tm.predict(np.array([[0,0,1,0,1,0,1,1,1,1,0,0]]))))
# print("Prediction: x1 = 1, x2 = 1, ... -> y = %d" % (tm.predict(np.array([[1,1,1,0,1,0,1,1,1,1,0,0]]))))
