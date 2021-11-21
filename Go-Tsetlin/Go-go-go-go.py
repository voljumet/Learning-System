import pickle
import string
from time import time
import numpy as np
from datetime import datetime

from sklearn.model_selection import train_test_split

from pyTsetlinMachine.tm import MultiClassConvolutionalTsetlinMachine2D

from process_go_ds import create_TM_representations


def timer():
    return datetime.now().strftime('%H:%M:%S, %d-%m')


def store_as_data(data, filename):
	fw = open(filename+'.data', 'wb')
	pickle.dump(data, fw)
	fw.close()


def load_data(filename):
	input_file = filename+'.data'
	fd = open(input_file, 'rb')
	return pickle.load(fd)


''' ----------------- SETTINGS ----------------------------------- '''
game_size = 9
sgf_dir = 'go9'

# Use "win", "black", or "white" to pick prediction model.


# True = Load data from folders and save as data-file.
# False = Load data from data-file.

while True:
	moves_to_predict = int(input("How many moves to predict?: "))
	if 5 > moves_to_predict > 2 or moves_to_predict == 0:
		break

while True:
	type_predict = input("What to prodict?: (win / black / white) ")
	if type_predict == "black" or type_predict == "white" or type_predict == "win":
		break

while True:
	inn = input("Load data from folder = T, from datafile = F: ")
	if inn == "T":
		newData = True
		break
	elif inn == "F":
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
	store_as_data(X_temp, "X_temp_"+type_predict+"_"+sgf_dir+"_"+str(game_size)+"_moves"+str(moves_to_predict))
	store_as_data(Y_temp, "Y_temp_"+type_predict+"_"+sgf_dir+"_"+str(game_size)+"_moves"+str(moves_to_predict))
else:
	print("Skipping loading from folder, using data-files ... \n")


X_temp = load_data("X_temp_"+type_predict+"_"+sgf_dir+"_"+str(game_size)+"_moves"+str(moves_to_predict))
Y_temp = load_data("Y_temp_"+type_predict+"_"+sgf_dir+"_"+str(game_size)+"_moves"+str(moves_to_predict))

# split into test / train and shuffle
X_train, X_test, Y_train, Y_test = train_test_split(X_temp, Y_temp, test_size=0.33, random_state=42, shuffle=True)

''' ------------- SETTINGS ------------- '''
clauses = 100
Threshold = 100
Forget_rate = 5
epochs = 250
''' ------------------------------------ '''
ctm = MultiClassConvolutionalTsetlinMachine2D(number_of_clauses=clauses, T=Threshold, s=Forget_rate, patch_dim=(9, 18), boost_true_positive_feedback=0)

print(f"\nPredict: {type_predict}, "+f"Number_of_clauses = {clauses}, "+ f"T = {Threshold}, "+ f"S = {Forget_rate}, "+ f"Started = {timer()}"+"\n")

f = open("log.txt", "a")
f.write(f"\nPredict: {type_predict}, "+f"Number_of_clauses = {clauses}, "+ f"T = {Threshold}, "+ f"S = {Forget_rate}, "+ f"Started = {timer()}"+"\n")
f.close()


results = np.zeros(0)
print("Training...")
for i in range(100):
	start = time()
	ctm.fit(X_train, Y_train, epochs=epochs)
	stop = time()

	results = np.append(results, 100*(ctm.predict(X_test) == Y_test).mean())
	print("#%d Mean Accuracy (%%): %.2f; Std.dev.: %.2f; Training Time: %.1f ms/epoch; Timestamp: %s" % (
		i+1, np.mean(results), np.std(results), (stop-start)/epochs, timer()))

	f = open("log.txt", "a")
	f.write("#%d Mean Accuracy (%%): %.2f; Std.dev.: %.2f; Training Time: %.1f ms/epoch; Timestamp: %s" % (
		i + 1, np.mean(results), np.std(results), (stop - start), timer()) + "\n")
	f.close()