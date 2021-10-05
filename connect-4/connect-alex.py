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

tm = MultiClassTsetlinMachine(1000, 40, 10.0)

print("\nAccuracy over 250 epochs:\n")
for i in range(250):
    start_training = time()
    tm.fit(X_train, Y_train, epochs=1, incremental=True)
    stop_training = time()

    start_testing = time()
    result = 100*(tm.predict(X_test) == Y_test).mean()
    stop_testing = time()

    print("#%d Accuracy: %.2f%% Training: %.2fs Testing: %.2fs" % (i+1, result, stop_training-start_training, stop_testing-start_testing))

# np.random.shuffle(training_dataset)
