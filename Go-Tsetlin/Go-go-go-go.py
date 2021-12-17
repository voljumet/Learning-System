import pickle
import string
from time import time
import numpy as np
from datetime import datetime

from sklearn.model_selection import train_test_split

from process_go_ds import create_TM_representations

while True:
	cuda = input("CUDA? (T / F): ")
	if cuda == "T":
		from PyTsetlinMachineCUDA.tm import MultiClassConvolutionalTsetlinMachine2D
		break
	elif cuda == "F":
		from pyTsetlinMachine.tm import MultiClassConvolutionalTsetlinMachine2D
		break


def timer():
	return datetime.now().strftime('%H:%M:%S, %d-%m')


def store_as_data(data, filename):
	fw = open(filename + '.data', 'wb')
	pickle.dump(data, fw)
	fw.close()


def load_data(filename):
	input_file = filename + '.data'
	fd = open(input_file, 'rb')
	return pickle.load(fd)


''' ----------------- SETTINGS ----------------------------------- '''
game_size = 9
sgf_dir = 'go9'


while True:
	moves_to_predict = int(input("How many moves used to predict? (B=odd numbers, W=even numbers| 0 will use ALL recursively):"))
	if 5 > moves_to_predict > 2 or moves_to_predict == 0:
		break

while True:
	type_predict = input("What to prodict?: (win / black / white) ")
	if type_predict == "black" or type_predict == "white" or type_predict == "win":
		break

while True:
	inn = input("Load data from folder = F, from datafile = D: ")
	if inn == "F":
		newData = True
		break
	elif inn == "D":
		newData = False
		break
''' -------------------------------------------------------------- '''

if newData:
	print("Dataset is being loaded from folder... \n")
	X_linear, Y_temp = create_TM_representations(sgf_dir, game_size, type_predict, moves_to_predict)
	print("Load complete ... \n")

	print("Reshaping from linear to matrix ... \n")
	X_temp = X_linear[:, :].reshape(X_linear.shape[0], 9, 18)
	print("Reshape complete ... \n")

	print("Saving as data-files ... \n")
	store_as_data(X_temp,
	              "X_temp_" + type_predict + "_" + sgf_dir + "_" + str(game_size) + "_moves" + str(moves_to_predict))
	store_as_data(Y_temp,
	              "Y_temp_" + type_predict + "_" + sgf_dir + "_" + str(game_size) + "_moves" + str(moves_to_predict))
else:
	print("Skipping loading from folder, using data-files ... \n")

X_temp = load_data("X_temp_" + type_predict + "_" + sgf_dir + "_" + str(game_size) + "_moves" + str(moves_to_predict))
Y_temp = load_data("Y_temp_" + type_predict + "_" + sgf_dir + "_" + str(game_size) + "_moves" + str(moves_to_predict))

# split into test / train and shuffle
X_train, X_test, Y_train, Y_test = train_test_split(X_temp, Y_temp, test_size=0.2, random_state=42, shuffle=True)

''' ------------- SETTINGS ------------- '''
clauses = 200
Threshold = 300
Forget_rate = 10
epochs = 250
''' ------------------------------------ '''
if cuda == "T":
	ctm = MultiClassConvolutionalTsetlinMachine2D(number_of_clauses=clauses, T=Threshold, s=Forget_rate, patch_dim=(9, 18), max_weight=50)
elif cuda == "F":
	ctm = MultiClassConvolutionalTsetlinMachine2D(number_of_clauses=clauses, T=Threshold, s=Forget_rate, patch_dim=(9, 18), boost_true_positive_feedback=0)

print(
	f"\nPredict: {type_predict}, " + f"Number_of_clauses = {clauses}, " + f"T = {Threshold}, " + f"S = {Forget_rate}, "
	+ f"Epocs = {epochs}, " + f"Started = {timer()}" + "\n")

f = open("log.txt", "a")
f.write(
	f"\nPredict: {type_predict}, " + f"Number_of_clauses = {clauses}, " + f"T = {Threshold}, " + f"S = {Forget_rate}, "
	+ f"Epocs = {epochs}, " + f"Started = {timer()}" + "\n")
f.close()


def train(epochs):
	results = np.zeros(0)
	print("Training...")
	for i in range(epochs):
		start = time()
		ctm.fit(X_train, Y_train, epochs=epochs)
		stop = time()

		results = np.append(results, 100 * (ctm.predict(X_test) == Y_test).mean())
		print("#%d Mean Accuracy (%%): %.2f; Std.dev.: %.2f; Training Time: %.1f ms/epoch; Timestamp: %s" % (
			i + 1, np.mean(results), np.std(results), (stop - start) / epochs, timer()))

		f = open("log.txt", "a")
		f.write("#%d Mean Accuracy (%%): %.2f; Std.dev.: %.2f; Training Time: %.1f ms/epoch; Timestamp: %s" % (
			i + 1, np.mean(results), np.std(results), (stop - start), timer()) + "\n")
		f.close()


train(epochs=100)

