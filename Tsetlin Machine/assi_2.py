from random import random
from random import choice

patients = [
    #2
    {'ge40': False, 'lt40': True, 'premeno': False, '3-5': False, '0-2': True, '6-8': False, '3': True, '2': False,
     '1': False},
    #4
    {'ge40': True, 'lt40': False, 'premeno': False, '3-5': False, '0-2': True, '6-8': False, '3': False, '2': True,
     '1': False},
    #6
    {'ge40': False, 'lt40': False, 'premeno': True, '3-5': False, '0-2': True, '6-8': False, '3': False, '2': False,
     '1': True},
    #1
    {'ge40': True, 'lt40': False, 'premeno': False, '3-5': True, '0-2': False, '6-8': False, '3': True, '2': False,
     '1': False},
    #3
    {'ge40': True, 'lt40': False, 'premeno': False, '3-5': False, '0-2': False, '6-8': True, '3': True, '2': False,
     '1': False},
    #5
    {'ge40': False, 'lt40': False, 'premeno': True, '3-5': False, '0-2': True, '6-8': False, '3': True, '2': False,
     '1': False}
]

non_recurring = patients[:3]
recurring_cancer = patients[3:]

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


# example_condition = ['ge40', '3-5', '2']
# print(evaluate_condition(recurring_cancer[0], example_condition))


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

# recurring_rule.get_condition()
# print("IF " + " AND ".join(recurring_rule.get_condition()) + " THEN Recurrence")


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


# for i in range(100):
#     observation_id = choice([0, 1, 2])
#     type_i_feedback(recurring_cancer[observation_id], recurring_rule)
# print("IF " + " AND ".join(recurring_rule.get_condition()) + " THEN Recurrence")


def type_ii_feedback(observation, memory):
    if evaluate_condition(observation, memory.get_condition()) == True:
        for feature in observation:
            if observation[feature] == False:
                memory.memorize_always(feature)
            elif observation[feature] == True:
                memory.memorize_always('NOT ' + feature)



# print("IF " + " AND ".join(recurring_rule.get_condition()) + " THEN Recurrence")
#
# print(evaluate_condition(non_recurring[0], recurring_rule.get_condition()))

# for i in range(100):
#     observation_id = choice([0, 1, 2])
#     type_ii_feedback(non_recurring[observation_id], recurring_rule)
# print(recurring_rule.get_memory())

# print("IF " + " AND ".join(recurring_rule.get_condition()) + " THEN Recurrence")
#
# print(evaluate_condition(non_recurring[0], recurring_rule.get_condition()))

def makeRule(inputRule, printa, epocs):
    for i in range(epocs):
        observation_id = choice([0, 1, 2])
        recurring = choice([0, 1])
        one = 0
        two = 0
        if printa == "recurring":
            one = recurring_cancer
            two = non_recurring
        else:
            one = non_recurring
            two = recurring_cancer

        if recurring == 1:
            type_i_feedback(one[observation_id], inputRule)
        else:
            type_ii_feedback(two[observation_id], inputRule)

    print("IF " + " AND ".join(inputRule.get_condition()) + " THEN ",printa)
    print(inputRule.get_memory())


def classify(observation, recurring_rules, non_recurring_rules, input):
    vote_sum = 0
    if input == "recurring":
        for recurring_rule in recurring_rules:
            if evaluate_condition(observation, recurring_rule.get_condition()) == True:
                vote_sum += 1
        for non_recurring_rule in non_recurring_rules:
            if evaluate_condition(observation, non_recurring_rule.get_condition()) == True:
                vote_sum -= 1
        if vote_sum >= 0:
            return "Recurrence"
        else:
            return "non-Recurrence"
    else:
        for recurring_rule in recurring_rules:
            if evaluate_condition(observation, recurring_rule.get_condition()) == True:
                vote_sum -= 1
        for non_recurring_rule in non_recurring_rules:
            if evaluate_condition(observation, non_recurring_rule.get_condition()) == True:
                vote_sum += 1
        if vote_sum >= 0:
            return "non-Recurrence"
        else:
            return "Recurrence"

# initialize the two rules
recurring_rule = Memory(0.9, 0.1,{'ge40': 5, 'lt40': 5, 'premeno': 5, 'NOT ge40': 5, 'NOT lt40': 5, 'NOT premeno': 5,
                            '3-5': 5, '0-2': 5, '6-8': 5,'NOT 3-5': 5, 'NOT 0-2': 5, 'NOT 6-8': 5,
                            '3': 5, '2': 5, '1': 5, 'NOT 3': 5, 'NOT 2': 5, 'NOT 1': 5})

non_recurring_rule = Memory(0.8, 0.2,{'ge40': 5, 'lt40': 5, 'premeno': 5, 'NOT ge40': 5, 'NOT lt40': 5, 'NOT premeno': 5,
                            '3-5': 5, '0-2': 5, '6-8': 5,'NOT 3-5': 5, 'NOT 0-2': 5, 'NOT 6-8': 5,
                            '3': 5, '2': 5, '1': 5, 'NOT 3': 5, 'NOT 2': 5, 'NOT 1': 5})

# print("IF " + " AND ".join(non_recurring_rule.get_condition()) + " THEN Non-Recurrence")

# classify(non_recurring[1], [recurring_rule], [non_recurring_rule])
#
# classify(recurring_cancer[2], [recurring_rule], [non_recurring_rule])

# This creates a rule -------------------------------------------------------------------------------------------------
makeRule(recurring_rule, "recurring", epocs=1000)

for eachPatient in patients:
    print(eachPatient, " has ", classify(eachPatient, [recurring_rule], [non_recurring_rule], "recurring"))

# initialize the recurring rule again
# recurring_rule = Memory(0.9, 0.1,{'ge40': 5, 'lt40': 5, 'premeno': 5, 'NOT ge40': 5, 'NOT lt40': 5, 'NOT premeno': 5,
#                             '3-5': 5, '0-2': 5, '6-8': 5,'NOT 3-5': 5, 'NOT 0-2': 5, 'NOT 6-8': 5,
#                             '3': 5, '2': 5, '1': 5, 'NOT 3': 5, 'NOT 2': 5, 'NOT 1': 5})

# This creates a rule -------------------------------------------------------------------------------------------------
makeRule(non_recurring_rule,"non-recurring", epocs=1000)

for eachPatient in patients:
    print(eachPatient, " has ", classify(eachPatient, [recurring_rule], [non_recurring_rule], "non-recurring"))
