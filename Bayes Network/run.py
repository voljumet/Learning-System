# Import required packages
import math
from pomegranate import *

# Initially the door selected by the guest is completely random
# A for rain and B for Sprinkler
rain = DiscreteDistribution({'T': 1. / 2, 'F': 1. / 2})

# The door containing the prize is also a random process
sprinkler = DiscreteDistribution({'T': 1. / 2, 'F': 1. / 2})

# The door Monty picks, depends on the choice of the guest and the prize door
watson = ConditionalProbabilityTable(
      [['T', 0.9],
       ['F', 0.1]], [rain])

holmes = ConditionalProbabilityTable(
      [['T', 'T', 1.0],
       ['T', 'F', 0.5],
       ['F', 'T', 0.5],
       ['F', 'F', 0.1]], [rain, sprinkler])


d1 = State(rain, name="rain")
d2 = State(sprinkler, name="sprinkler")
d3 = State(watson, name="watson")
d4 = State(holmes, name="holmes")

# Building the Bayesian Network
network = BayesianNetwork("Solving rainy garden With Bayesian Networks")
network.add_states(d1, d2, d3, d4)
network.add_edge(d1, d3)
network.add_edge(d1, d4)
network.add_edge(d2, d4)
network.bake()

beliefs = network.predict_proba({'rain': 'T'})
beliefs = map(str, beliefs)
print("n".join("{}t{}".format(state.name, belief) for state, belief in zip(network.states, beliefs)))
