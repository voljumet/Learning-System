import pickle
import string
from time import time
import numpy as np

from sklearn.model_selection import train_test_split

from PyTsetlinMachineCUDA.tm import MultiClassConvolutionalTsetlinMachine2D
from process_go_ds import create_TM_representations


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
type_predict = "black"
inn = ''

# True = Load data from folders and save as data-file.
# False = Load data from data-file.
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
	X_linear, Y_temp = create_TM_representations(sgf_dir, game_size, type_predict)
	print("Load complete ... \n")

	print("Reshaping from linear to matrix ... \n")
	X_temp = X_linear[:, :].reshape(X_linear.shape[0], 9, 18)
	print("Reshape complete ... \n")

	print("Saving as data-files ... \n")
	store_as_data(X_temp, "X_temp_"+type_predict+"_"+sgf_dir+"_"+str(game_size))
	store_as_data(Y_temp, "Y_temp_"+type_predict+"_"+sgf_dir+"_"+str(game_size))
else:
	print("Skipping loading from folder, using data-files ... \n")


X_temp = load_data("X_temp_"+type_predict+"_"+sgf_dir+"_"+str(game_size))
Y_temp = load_data("Y_temp_"+type_predict+"_"+sgf_dir+"_"+str(game_size))

# split into test / train and shuffle
X_train, X_test, Y_train, Y_test = train_test_split(X_temp, Y_temp, test_size=0.33, random_state=42, shuffle=True)

''' ----------------- SETTINGS ----------------------------------- '''
clauses = 200
Threshold = 200
Forget_rate = 50
epochs = 100
ctm = MultiClassConvolutionalTsetlinMachine2D(number_of_clauses=clauses, T=Threshold, s=Forget_rate, patch_dim=(9, 18), max_weight=16)

f = open("log.txt", "a")
f.write(f"Predict: {type_predict}, "+f"Number_of_clauses = {clauses}, "+ f"T = {Threshold}, "+ f"S = {Forget_rate}"+ f"Epocs = {epochs}" + "\n")
f.close()


results = np.zeros(0)
print("Training...")
for i in range(100):
	start = time()
	ctm.fit(X_train, Y_train, epochs=epochs)
	stop = time()

	results = np.append(results, 100*(ctm.predict(X_test) == Y_test).mean())
	print("#%d Mean Accuracy (%%): %.2f; Std.dev.: %.2f; Training Time: %.1f ms/epoch" % (i+1, np.mean(results), np.std(results), (stop-start)/epochs))

	f = open("log.txt", "a")
	f.write("#%d Mean Accuracy (%%): %.2f; Std.dev.: %.2f; Training Time: %.1f ms/epoch" % (i + 1, np.mean(results), np.std(results), (stop - start)) + "\n")
	f.close()