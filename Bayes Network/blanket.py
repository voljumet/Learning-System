import numpy as np


def ran():
    return round(np.random.uniform(0,1),2)


def yielda(input_list, remove):
    list = []
    for each in input_list:
        if each != remove:
            list.append(each)
    return list


def expand(inputt):
    lista = []
    for each in inputt:
        lista.append(each)
        lista.append((1 - each).__round__(2))
    return lista


def truuy(bool):
    if bool:
        return 0
    else:
        return 1

# Estimate P(Rain | Holmes' Grass is Wet, Wasons grass is Wet)
# P(Rain | Holmes' Grass is Wet, Wasons grass is Wet) = P(Rain | Holmes' Grass is Wet) * P(Wasons grass is Wet | Holmes' Grass is Wet)
# P(Rain | H)

# Implement data structure for Wet Grass Bayesian Network (slide 7),  including graph and probability tables
# Assign appropriate probabilities to the probability tables
# Implement Stochastic Simulation inference scheme
# Estimate P(Rain | Holmes’ Grass is Wet, Watson’s  Grass is Wet) and 3 other queries you select.


nodes = {"Rain": {"Parents": [], "Children": ["Watson", "Holmes"], "ParentOfChild": ["Sprinkler"]},
         "Sprinkler": {"Parents": [], "Children": ["Holmes"], "ParentOfChild": ["Rain"]},
         "Watson": {"Parents": ["Rain"], "Children": [], "ParentOfChild": []},
         "Holmes": {"Parents": ["Rain", "Sprinkler"], "Children": [], "ParentOfChild": []}}


# Creates a table with random values
def create_table():
    return {"Rain": [ran()],
        "Sprinkler": [ran()],
        "Watson": [ran(), ran()],
        "Holmes": [ran(), ran(), ran(), ran()]}


def ikke(inn):
    return 1 - inn


def network(node, node_bool, given_nodes, given_nodes_p, given_bool_1, given_bool_2, graph, table, value):
    for each_giv in given_nodes:
        table[each_giv][0] = given_nodes_p
    table[node][0] = value
    prob = []
    undersiden = []
    prob.append(table[node][truuy(node_bool)])

    for each_given in given_nodes:
        if graph[node]["Children"].__contains__(each_given):
            if len(graph[each_given]["Parents"]) > 1:
                n1, n2 = 2, 1
                if node == "Rain":
                    n1, n2 = 1, 2
                if node_bool:
                    prob.append(table[each_given][0] + table[each_given][n1])
                    undersiden.append(table[each_given][0] + table[each_given][n1])
                else:
                    prob.append(table[each_given][n2] + table[each_given][3])
                    undersiden.append(table[each_given][n2] + table[each_given][3])
            else:
                prob.append(table[each_given][truuy(node_bool)])

                undersiden.append(table[each_given][truuy(node_bool)] * table[node][truuy(node_bool)] +
                                  table[each_given][ikke(truuy(node_bool))] * ikke(table[node][truuy(node_bool)]))


    if graph[node]["Parents"].__contains__(given_nodes):
        if given_bool_1:
            if given_bool_2:
                return table[node][0]
            else:
                return table[node][1]
        else:
            if given_bool_2:
                return table[node][2]
            else:
                return table[node][3]

    summ = 1
    summm = 1
    for each in prob:
        if each == 0.0:
            each = 0.00001
        summ *= each

    for each in undersiden:
        if each == 0.0:
            each = 0.00001
        summm *= each
    return summ / summm


table = create_table()
# ut = network(node="Rain", node_bool=True, given_nodes=["Watson"], given_nodes_p=[0.9],
#                  given_bool_1=True, given_bool_2=True, graph=nodes, table=table)


# def new(pr,p_w_g_r,p_h_g_r_s1,p_h_g_r_s2):
#     return pr * p_w_g_r * (p_h_g_r_s1 + p_h_g_r_s2)
#
# two = new(table["Rain"][0],table["Watson"][0],table["Holmes"][0],table["Holmes"][1])

def epoc(epocs):
    list = []
    value = ran()
    for i in range(epocs):

        node = "Rain"
        value = network(node=node, node_bool=True, given_nodes=["Watson","Holmes"], given_nodes_p=0.90,given_bool_1=True,
                            given_bool_2=True, graph=nodes, table=table, value=value)
        list.append(value)
    print("Probability of", node, "is T: ", sum(list)/len(list))


epoc(5000)

''' ------------------------------------------------------------------------------
Will always be 50% when everything is random? choose variable(s) to not be random? 
-------------------------------------------------------------------------------'''

    # if graph[node]["Children"].__contains__(given_node_1):
    #     if len(graph[])


    # for each_c in graph[node]["Children"]:                    # får holmes og watson
    #     for child_value in table[each_c]:                     # each_c = watson, child_
    #         if given_bool_2 == True:
    #             child0 += table[given_node_2][0]
            # if graph[child_value]["Parent"] > 0:


    #         sum_A = 0
    #         if len(graph[each_poc]["children"]) > 0:
    #             for each_child in graph[each_poc]["children"]:
    #                 if len(graph[each_child]["children"]) > 0:
    #                     for each_children in graph[each_child]["children"]:
    #                         sum_A += (table[each_children])
    #
    #
    # for E in E_table:
    #     sum_A = 0
    #     for i in range(len(A_table)):
    #         sum_A += table_a[i] * table_m[i] * table_j[i]
    #     sum += E * sum_A
    # return (node * sum) / undersiden







# if graph[node] == "Parents":
#
#     pass
#     # gjør noe
# elif graph[node] == "children":
#     pass
#     # gjør noe
# elif graph[node] == "ParentOfChild":
#     pass
    # gjør noe
    # probability = table["Rain"] * table["Holmes"][0] * table["Rain"] * table["Sprinkler"] + \
    # table["Holmes"][1] * table["Rain"] * (1 - table["Sprinkler"]) + \
    # table["Holmes"][2] * nott(table["Rain"]) * table["Sprinkler"] + \
    # table["Holmes"][3] * nott(table["Rain"]) * nott(table["Sprinkler"]) + \
    # table["Watson"][0] * table["Rain"] + table["Watson"][1] * nott(table["Rain"])
    # print(probability)
