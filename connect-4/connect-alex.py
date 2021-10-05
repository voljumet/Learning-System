from pyTsetlinMachine.tm import MultiClassTsetlinMachine
import numpy as np
from time import time


games = np.zeros((67557, 7, 6, 3))

filename = "connect-4.data"
y_data = np.zeros(67557)
gc = 0
yc = 0
xc = 0
var = 0
with open(filename) as my_file:
    for line in my_file:
        sp = line.split(",")
        for splittet in sp:
            if xc == 6:
                yc += 1
                xc = 0
            elif yc == 7:
                yc = 0

            if splittet == 'b':
                var = 0
            elif splittet == 'x':
                var = 1
            elif splittet == 'o':
                var = 2
            else:
                if splittet == 'win\n':
                    var = 1
                elif splittet == 'loss\n':
                    var = 2
                else:
                    var = 3
                y_data[gc] = var
                continue
            # print(f'gc{gc},y{yc},xc{xc},var{var},split{splittet}')
            games[gc][yc][xc][var] = 1
            xc += 1
        gc += 1

print("Done loading", y_data)

X_train = games[:54000]
Y_train = y_data[:54000]

X_test = games[54000:]
Y_test = y_data[54000:]

X_train = X_train.reshape((X_train.shape[0], 7*6*3))
X_test = X_test.reshape((X_test.shape[0], 7*6*3))

# Y_train = Y_train.reshape((Y_train.shape[0], 1))
# Y_test = Y_test.reshape((Y_test.shape[0], 1))



def kjor(p,j,k,rangez):
    tm = MultiClassTsetlinMachine(p, j, k)
    print(f"\nT: {j}, s: {k}, #c: {p}")
    # print(f"Accuracy over {rangez} epochs:")
    results = []
    for i in range(rangez):
        # start_training = time()
        tm.fit(X_train, Y_train, epochs=1, incremental=True)
        # stop_training = time()

        # start_testing = time()
        results.append(100*(tm.predict(X_test) == Y_test).mean())
        # stop_testing = time()
    results.sort(reverse=True)
    print(f"Best Accuracy over {rangez} epochs: {results[0]}")
                                                                       # , stop_training-start_training, stop_testing-start_testing))

# np.random.shuffle(training_dataset)
for i in range(100,3000,100):
    print("yes")
    for j in range(10, 200, 5):
        for k in range(3, 100, 3):
            kjor(i, j, k, 6)