
cars = [
    {'Four Wheels': True, 'Transports People': True, 'Wings': False, 'Yellow': False, 'Blue': True},
    {'Four Wheels': True, 'Transports People': True, 'Wings': False, 'Yellow': True, 'Blue': False},
    {'Four Wheels': True, 'Transports People': True, 'Wings': False, 'Yellow': True, 'Blue': False}
]


def evaluate_condition(observation, condition):
    truth_value_of_condition = True
    for feature in observation:
        if feature in condition and observation[feature] == False:
            truth_value_of_condition = False
            break
        if 'NOT ' + feature in condition and observation[feature] == True:
            truth_value_of_condition = False
            break
    return truth_value_of_condition


example_condition = ['Four Wheels', 'Transports People', 'NOT Wings']

evaluate_condition(cars[0], example_condition)



class Memory:
    def __init__(self, forget_value, memorize_value, memory):
        self.memory = memory
        self.forget_value = forget_value
        self.memorize_value = memorize_value

    def get_memory(self):
        return self.memory

    def get_literals(self):
        return list(self.memory.keys())

    def get_condition(self):
        condition = []
        for literal in self.memory:
            if self.memory[literal] >= 6:
                condition.append(literal)
        return condition

    def memorize(self, literal):
        if random() <= self.memorize_value and self.memory[literal] < 10:
            self.memory[literal] += 1

    def forget(self, literal):
        if random() <= self.forget_value and self.memory[literal] > 1:
            self.memory[literal] -= 1

    def memorize_always(self, literal):
        if self.memory[literal] < 10:
            self.memory[literal] += 1



car_rule = Memory(0.9, 0.1,
                  {'Four Wheels': 10, 'NOT Four Wheels': 2, 'Transports People': 9, 'NOT Transports People': 3,
                   'Wings': 1, 'NOT Wings': 5, 'Yellow': 1, 'NOT Yellow': 4, 'Blue': 6, 'NOT Blue': 4})

car_rule.get_condition()

print("IF " + " AND ".join(car_rule.get_condition()) + " THEN Car")

from random import random


car_rule.forget('Blue')
car_rule.forget('Blue')
print(car_rule.memory)

car_rule = Memory(0.9, 0.1, {'Four Wheels': 5, 'NOT Four Wheels': 5, 'Transports People': 5, 'NOT Transports People': 5,
                             'Wings': 5, 'NOT Wings': 5, 'Yellow': 5, 'NOT Yellow': 5, 'Blue': 5, 'NOT Blue': 5})



def type_i_feedback(observation, memory):
    remaining_literals = memory.get_literals()
    if evaluate_condition(observation, memory.get_condition()) == True:
        for feature in observation:
            if observation[feature] == True:
                memory.memorize(feature)
                remaining_literals.remove(feature)
            elif observation[feature] == False:
                memory.memorize('NOT ' + feature)
                remaining_literals.remove('NOT ' + feature)
    for literal in remaining_literals:
        memory.forget(literal)



from random import choice

for i in range(100):
    observation_id = choice([0, 1, 2])
    type_i_feedback(cars[observation_id], car_rule)

print("IF " + " AND ".join(car_rule.get_condition()) + " THEN Car")

planes = [
    {'Four Wheels': True, 'Transports People': True, 'Wings': True, 'Yellow': False, 'Blue': True},
    {'Four Wheels': True, 'Transports People': False, 'Wings': True, 'Yellow': True, 'Blue': False},
    {'Four Wheels': False, 'Transports People': True, 'Wings': True, 'Yellow': False, 'Blue': True}
]



def type_ii_feedback(observation, memory):
    if evaluate_condition(observation, memory.get_condition()) == True:
        for feature in observation:
            if observation[feature] == False:
                memory.memorize_always(feature)
            elif observation[feature] == True:
                memory.memorize_always('NOT ' + feature)



car_rule = Memory(0.9, 0.1,
                  {'Four Wheels': 10, 'NOT Four Wheels': 1, 'Transports People': 10, 'NOT Transports People': 1,
                   'Wings': 1, 'NOT Wings': 1, 'Yellow': 1, 'NOT Yellow': 1, 'Blue': 1, 'NOT Blue': 1})
print("IF " + " AND ".join(car_rule.get_condition()) + " THEN Car")

print(evaluate_condition(planes[0], car_rule.get_condition()))

for i in range(100):
    observation_id = choice([0, 1, 2])
    type_ii_feedback(planes[observation_id], car_rule)
print(car_rule.get_memory())
{'Four Wheels': 10, 'NOT Four Wheels': 6, 'Transports People': 10, 'NOT Transports People': 6, 'Wings': 1,
 'NOT Wings': 6, 'Yellow': 6, 'NOT Yellow': 1, 'Blue': 1, 'NOT Blue': 6}

print("IF " + " AND ".join(car_rule.get_condition()) + " THEN Car")

print(evaluate_condition(planes[0], car_rule.get_condition()))

for i in range(100):
    observation_id = choice([0, 1, 2])
    car = choice([0, 1])
    if car == 1:
        type_i_feedback(cars[observation_id], car_rule)
    else:
        type_ii_feedback(planes[observation_id], car_rule)

print("IF " + " AND ".join(car_rule.get_condition()) + " THEN Car")

print(car_rule.get_memory())
{'Four Wheels': 10, 'NOT Four Wheels': 1, 'Transports People': 10, 'NOT Transports People': 1, 'Wings': 1,
 'NOT Wings': 10, 'Yellow': 1, 'NOT Yellow': 1, 'Blue': 1, 'NOT Blue': 1}



def classify(observation, car_rules, plane_rules):
    vote_sum = 0
    for car_rule in car_rules:
        if evaluate_condition(observation, car_rule.get_condition()) == True:
            vote_sum += 1
    for plane_rule in plane_rules:
        if evaluate_condition(observation, plane_rule.get_condition()) == True:
            vote_sum -= 1
    if vote_sum >= 0:
        return "Car"
    else:
        return "Plane"



plane_rule = Memory(0.9, 0.1,
                    {'Four Wheels': 1, 'NOT Four Wheels': 1, 'Transports People': 1, 'NOT Transports People': 1,
                     'Wings': 10, 'NOT Wings': 1, 'Yellow': 1, 'NOT Yellow': 1, 'Blue': 1, 'NOT Blue': 1})
print("IF " + " AND ".join(plane_rule.get_condition()) + " THEN Plane")

classify(planes[1], [car_rule], [plane_rule])

classify(cars[2], [car_rule], [plane_rule])
