from sklearn.datasets import fetch_20newsgroups
import numpy as np
import enchant
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from collections import defaultdict
import pickle

# # load english dictionary
# d = enchant.Dict("en_US")

# use the stem of the word
ps = PorterStemmer()

lemmatizer = WordNetLemmatizer()

newsgroups_train = fetch_20newsgroups(subset='train', remove=['header','footer'])
newsgroups_test = fetch_20newsgroups(subset='test', remove=['header','footer'])


# read the stop_words file
a_file = open("stop_words.txt", "r")

stop_words = []
for line in a_file:
  stripped_line = line.strip()
  stop_words.append(stripped_line)
a_file.close()


# Training datasets
y_train_temp = newsgroups_train.target[:]
x_train_temp = newsgroups_train.data[:]

#Testing datasets
y_test_temp = newsgroups_test.target[:]
x_test_temp = newsgroups_test.data[:]

# All the categories
categories = newsgroups_train.target_names


def split_words(text_to_split):
    vocabulary = defaultdict(int)
    posts = defaultdict(list)
    for count, each_text in enumerate(text_to_split):
        for word in each_text.lower().split():
            if word not in stop_words:
                word = lemmatizer.lemmatize(word)
                posts[categories[y_train_temp[count]]].append(word)
                vocabulary[word] += 1
    return vocabulary, posts


def store_as_data(data, filename):
    fw = open(filename+'.data', 'wb')
    pickle.dump(data, fw)
    fw.close()

def load_data(filename):
    inputFile = filename+'.data'
    fd = open(inputFile, 'rb')
    return pickle.load(fd)

def x_cleanup():
    posts_test = []
    for text in x_test_temp:
        words = []
        for word in text.lower().split():
            words.append(lemmatizer.lemmatize(word))
        posts_test.append(words)
    return posts_test



''' Use when new data is available set to True else set to False '''

newData = True

''' ------------------------------- '''

if newData:
    print("Dataset is being cleaned up ... \n")
    vocabulary, posts = split_words(x_train_temp)
    print(len(vocabulary))

    print("Cleanup complete ... \n")
    print("Saving data as file ... \n")
    store_as_data(vocabulary, "vocabulary")
    store_as_data(posts, "posts")

    store_as_data(x_cleanup(), "posts_test")
    store_as_data(y_test_temp, "y_train_temp")
else:
    print("Skipping data cleanup ... \n")

print("Loading data from file ... \n")

''' Data is loaded from file, be sure that your data files holds the correct data '''

posts = load_data("posts")
vocabulary = load_data("vocabulary")
print("Vocabulary size:", len(vocabulary))

post_to_be_classified = load_data("posts_test")
y_test_temp = load_data("y_train_temp")

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

print("Probability calculated ... \n")


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


# Finds group with max P(O | H) * P(H)

p_group = {}

# calculates probability of each category based on the number of posts
for category_number, each_category in enumerate(categories):
    p_group[each_category] = number_of_texts_in_category[category_number] / total_number_of_texts

print("Predicting categories from testdata ... \n")

accuracy = 0
for i in range(len(x_test_temp)):
    max_group = 0
    max_p = 1
    # for count, each_post in enumerate(x_test_temp):
    for candidate_group in posts.keys():
    # Calculates P(O | H) * P(H) for candidate group
        p = np.log(p_group[candidate_group])
        for word in post_to_be_classified[i]:
            if word in vocabulary:
                p += np.log(p_word_given_group[candidate_group][word])
        # print("P: ", p, " candidate_group: ", candidate_group)

        if p > max_p or max_p == 1:
            max_p = p
            max_group = candidate_group
    if max_group == categories[y_test_temp[i]]:
        accuracy += 1
    # print("Category Pred:", max_group, " | G. Truth:", categories[y_test_temp[i]])
    # print(max_p)

print("Accuracy: ", accuracy/len(x_test_temp))
