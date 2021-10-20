from sklearn.datasets import fetch_20newsgroups
import numpy as np
from collections import defaultdict

newsgroups_train = fetch_20newsgroups(subset='train', remove=['headers', 'footers'])
newsgroups_test = fetch_20newsgroups(subset='test', remove=['headers', 'footers'])

# Training datasets
y_train_temp = newsgroups_train.target
x_train_temp = newsgroups_train.data

#Testing datasets
y_test_temp = newsgroups_test.target
x_test_temp = newsgroups_test.data

# All the categories
categories = newsgroups_train.target_names


# x_train = []
# dictionary = {}


def split_words(text_to_split):
    vocabulary = defaultdict(int)
    posts = defaultdict(list)
    for count, each_text in enumerate(text_to_split):
        # list_of_words = []
        for word in each_text.split():
            posts[categories[y_train_temp[count]]].append(word)
            vocabulary[word] += 1
    return vocabulary, posts

vocabulary, posts = split_words(x_train_temp)

p_word_given_group = {}
for group in posts.keys():

        p_word_given_group[group] = {}

        # Counts the number of words
        for word in vocabulary.keys():
            p_word_given_group[group][word] = 1.0
            ''' what is the point of this when its added below? '''

        for word in posts[group]:
            if word in vocabulary:
                p_word_given_group[group][word] += 1.0

        # Calculates probabilities
        for word in vocabulary.keys():
            p_word_given_group[group][word] /= len(posts[group]) + len(vocabulary)

def countFreq(arr):
    # Mark all array elements as not visited
    counter = {}
    n = len(arr)
    visited = [False for i in range(n)]
    # Traverse through array elements
    # and count frequencies
    for i in range(n):

        # Skip this element if already
        # processed
        if (visited[i] == True):
            continue

        # Count frequency
        count = 1
        for j in range(i + 1, n, 1):
            if (arr[i] == arr[j]):
                visited[j] = True
                count += 1

        counter[arr[i]] = count
        # print(arr[i], count)
    return counter


total_number_of_texts = len(y_train_temp)
number_of_texts_in_category = countFreq(y_train_temp)


print("wait a minute")

'''
testing value :
'''
test_value = 555

post_to_be_classified = x_test_temp[test_value].split()
# Finds group with max P(O | H) * P(H)
max_group = 0
max_p = 1
p_group = {}

# length = y_train_temp

for category_number, each_category in enumerate(categories):
    p_group[each_category] = number_of_texts_in_category[category_number] / total_number_of_texts

for candidate_group in posts.keys():
# Calculates P(O | H) * P(H) for candidate group
    p = np.log(p_group[candidate_group])
    for word in post_to_be_classified:
        if word in vocabulary:
            p += np.log(p_word_given_group[candidate_group][word])
    print("P: ", p, " candidate_group: ", candidate_group)

    if p > max_p or max_p == 1:
        max_p = p
        max_group = candidate_group

print("Predicted category :", max_group)
print("Ground truth category :", categories[y_test_temp[test_value]])

print("juice")





