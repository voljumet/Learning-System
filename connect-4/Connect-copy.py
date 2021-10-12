from pyTsetlinMachine.tm import MultiClassTsetlinMachine
# from pyTsetlinMachineCUDA import MultiClass

import numpy as np
from time import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

x_total = np.empty((67557, 42*2))
y_total = np.empty((67557))

filename = "connect-4.data"
win_count, lose_count, draw_count = 0, 0, 0
start_loading = time()
with open(filename) as my_file:
    for count, line in enumerate(my_file):
        bits_features = np.array([])
        each = line.split(",")
        if line.__contains__("win"):
            y_total[count] = 0
        elif line.__contains__("loss"):
            y_total[count] = 1
        else:
            y_total[count] = 2

        for feature in each:
            if feature == 'b':
                bits_features = np.append(bits_features, [0, 0])  # bit representation for b
            elif feature == 'x':
                bits_features = np.append(bits_features, [1, 0])  # bit representation for x
            elif feature == 'o':
                bits_features = np.append(bits_features, [0, 1])  # bit representation for o

        x_total[count] = bits_features

stop_loading = time()
print(f"Loaded after: {stop_loading-start_loading}ms")

# X_train = x_total[:60800]
# Y_train = y_total[:60800]

# X_test = x_total[60800:]
# Y_test = y_total[60800:]

X_train, X_test, Y_train, Y_test = train_test_split(x_total, y_total, random_state=42, shuffle=True, test_size=0.25)
# print(x_total)
# f = open("write.csv", "a")

start_training = time()
tm = MultiClassTsetlinMachine(1500, 20, 10)
stop_training = time()
print(f"time: {stop_training - start_training}")
Accuracy = []
print("\nAccuracy over 10 epochs:\n")
for i in range(100):
    start_training = time()
    tm.fit(X_train, Y_train, epochs=1, incremental=True)
    stop_training = time()

    start_testing = time()
    result = 100 * (tm.predict(X_test) == Y_test).mean()
    stop_testing = time()
    Accuracy.append(result)
    print("#%d Accuracy: %.2f%% Training: %.2fs Testing: %.2fs" % (
    i + 1, result, stop_training - start_training, stop_testing - start_testing))


clauses_to_inverstigate = 15
num_feat_to_investigate = 40

print("\nClass 0 Positive Clauses: \n")
for j in range(0, clauses_to_inverstigate, 2):
    print("Clause #%d:" % (j), end=' ')
    l = []
    for k in range(num_feat_to_investigate * 2):
        if tm.ta_action(0, j, k) == 1:
            if k < num_feat_to_investigate:
                l.append("x%d" % (k))
            else:
                l.append("¬x%d" % (k-num_feat_to_investigate))
    print("∧" .join(l))

print("\nClass 0 Negative Clauses: \n")
for j in range(1, clauses_to_inverstigate, 2):
    print("Clause #%d:" % (j), end='')
    l = []
    for k in range(num_feat_to_investigate * 2):
        if tm.ta_action(0, j, k) == 1:
            if k < num_feat_to_investigate:
                l.append("x%d" % (k))
            else:
                l.append("¬x%d" % (k-num_feat_to_investigate))
    print("∧" .join(l))

print("\nClass 1 Positive Clauses: \n")
for j in range(0, clauses_to_inverstigate, 2):
    print("Clause #%d:" % (j), end=' ')
    l = []
    for k in range(num_feat_to_investigate * 2):
        if tm.ta_action(1, j, k) == 1:
            if k < num_feat_to_investigate:
                l.append("x%d" % (k))
            else:
                l.append("¬x%d" % (k-num_feat_to_investigate))
    print("∧" .join(l))

print("\nClass 1 Negative Clauses: \n")
for j in range(1, clauses_to_inverstigate, 2):
    print("Clause #%d:" % (j), end='')
    l = []
    for k in range(num_feat_to_investigate * 2):
        if tm.ta_action(1, j, k) == 1:
            if k < num_feat_to_investigate:
                l.append("x%d" % (k))
            else:
                l.append("¬x%d" % (k-num_feat_to_investigate))
    print("∧" .join(l))


print("\nClass 2 Positive Clauses: \n")
for j in range(0, clauses_to_inverstigate, 2):
    print("Clause #%d:" % (j), end=' ')
    l = []
    for k in range(num_feat_to_investigate * 2):
        if tm.ta_action(2, j, k) == 1:
            if k < num_feat_to_investigate:
                l.append("x%d" % (k))
            else:
                l.append("¬x%d" % (k - num_feat_to_investigate))
    print("∧" .join(l))

print("\nClass 2 Negative Clauses: \n")
for j in range(1, clauses_to_inverstigate, 2):
    print("Clause #%d:" % (j), end='')
    l = []
    for k in range(num_feat_to_investigate * 2):
        if tm.ta_action(2, j, k) == 1:
            if k < num_feat_to_investigate:
                l.append("x%d" % (k))
            else:
                l.append("¬x%d" % (k-num_feat_to_investigate))
    print("∧" .join(l))

plt.plot(Accuracy)
plt.show()