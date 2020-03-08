import os
from os import path, getcwd, listdir
from nltk.corpus import stopwords
import collections
import time

PATH_TO_DATA = path.join(getcwd(), 'data/cs276')


# Index all file to be able to retrieve them
def create_file_index():
    file_index_dict = {}
    current_file_index = 0
    for directory_index in range(9):
        path_directory = path.join(PATH_TO_DATA, str(directory_index))
        for filename in listdir(path_directory):
            file_index_dict[current_file_index] = filename
            current_file_index += 1
    return file_index_dict


# Save file index
def save_file_index(file_index):
    with open('file_index.txt', 'w') as file:
        for index, filename in file_index.items():
            file.write("{} {}\n".format(index, filename))


# Load file index
def load_file_index(path_to_file_index):
    file_index_dict = {}
    with open(path_to_file_index, 'r') as file:
        for line in file.readlines():
            line = line.split(' ')
            file_index_dict[int(line[0])] = line[1].rstrip('\n')
    return file_index_dict


# Remove stopwords
# 1) Take the stopwords list from nltk
stopwords = stopwords.words('english')


def count_frequency(collection):
    tokens_count = collections.Counter()
    for list_tokens in collection.values():
        tokens_count.update(list_tokens)
    return tokens_count


def not_contains_digit(s):
    return all(not i.isdigit() for i in s)


def create_collection():
    current_file_index = 0
    collection = {}
    for i in range(9):
        path_directory = path.join(PATH_TO_DATA, str(i))
        for filename in listdir(path_directory):
            path_to_file = path.join(path_directory, filename)
            with open(path_to_file, 'r') as file:
                tokens = []
                for line in file.readlines():
                    tokens.extend(line.rstrip('\n').split(' '))
                filtered_tokens = filter(not_contains_digit, tokens)
            collection[current_file_index] = filtered_tokens
            current_file_index += 1
    return collection


# TODO : think about removing _, +, and other terms
# Evaluate time to create the collection
start = time.time()
standford_collection = create_collection()
frequencies = count_frequency(standford_collection)
end = time.time()
print(end - start)
# 40 sec pour le chargement

most_frequent = frequencies.most_common(30)
print(most_frequent)
