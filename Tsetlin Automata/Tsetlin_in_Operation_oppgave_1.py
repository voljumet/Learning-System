#!/usr/bin/python

import random
import math
import numpy as np
import matplotlib.pyplot as pl

class Environment:
    def __init__(self, penlty):
        self.penlty = penlty

    def penalty(self):
        if random.random() <= self.penlty:
            return True
        else:
            return False
       

class Tsetlin:
    def __init__(self, n):
        # n is the number of states per action
        self.n = n

        # Initial state selected randomly
        self.state = random.choice([self.n, self.n+1])

    def reward(self):
        if self.state <= self.n and self.state > 1:
            self.state -= 1
        elif self.state > self.n and self.state < 2 * self.n:
            self.state += 1

    def penalize(self):
        if self.state <= self.n:
            self.state += 1
        elif self.state > self.n:
            self.state -= 1

    def makeDecision(self):
        if self.state <= self.n:
            return 0
        else:
            return 1


def calculate_penalty(yesVotes):
    if yesVotes > correctVotes:
        return 1 - (0.6 - (yesVotes - 3) * 0.2)
    else:
        return 1 - (yesVotes * 0.2)


states = 30
players = 5
correctCutOff = 15
correctVotes = math.ceil(players/2)
performance = []
roundCounter = []


def game():
    TA = []
    for i in range(players):
        TA.append(Tsetlin(states))

    action_count = [0, 0]
    plot_results = ([], [])
    correctCounter = 0
    i = 0
    totalCorrect = 0
    while correctCounter < correctCutOff:
        actions = []
        for k in range(players):
            actions.append(TA[k].makeDecision())

        yesCount = np.sum(actions)
        # plot_results[0].append(i)
        plot_results[1].append(yesCount)
        env = Environment(calculate_penalty(yesCount))
        for j in range(len(actions)):
            penalty = env.penalty()
            # yesNo = ''
            # if actions[j] == 0:
            #     yesNo = 'No'
            # else:
            #     yesNo = 'Yes'
            # print(f"Player {j}, State {TA[j].state}, Vote:{yesNo}")
            action_count[actions[j]] += 1

            if penalty:
                TA[j].penalize()
            else:
                TA[j].reward()
        if correctVotes == plot_results[1][i]:
            correctCounter += 1
            totalCorrect += 1
        else:
            correctCounter = 0
        i += 1
    # print(f"Votes for No: {action_count[0]} Votes for Yes: {action_count[1]}")
    # print(f"Total correct: {totalCorrect} of {i}")
    # print(f"probability of correct = {totalCorrect/i}")
    roundCounter.append(i)
    performance.append(totalCorrect / i)


for i in range(1000):
    game()
print(f"average rounds: {np.sum(roundCounter) / len(roundCounter)}")
print(f"Performance: {np.sum(performance) / len(performance)}")
# hvor mange riktige per runde
# probability = match/ runder
# performance


# pl.plot(plot_results[0], plot_results[1])
pl.show()

# per action, more rewards over time, check penalties

