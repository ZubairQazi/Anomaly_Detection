import json
import random

import json
import zstandard as zstd

from collections import Counter

import torch
import numpy as np

from sklearn.utils import shuffle
import sparse

import pandas as pd
import re

from scipy.io import savemat
import os

def load_text_dataset_from_csv(path, subset_size=None):
    data = pd.read_csv(path)
    
    dataset = data.to_numpy()

    data, labels = dataset[:, 0], dataset[:, 1]
    data, labels = shuffle(data, labels)

    if subset_size:
        data, labels = data[:subset_size], labels[:subset_size]

    # Count the occurrences of all terms in the dataset
    term_counts = Counter()

    # Initialize an empty dictionary to store term indices
    term_indices = {}

    for idx, document in enumerate(data):
        try:
            # for term in document.split():
            for term in re.findall(r"(\w+|[^\w\s])", document):
                if term not in term_indices:
                    # Assign the next available index to the term
                    term_indices[term] = len(term_indices)
        except:
            print(f'Failed to process document #{idx}')
            continue

    # Get the number of unique terms
    num_unique_terms = len(term_indices)

    return data, labels, num_unique_terms, term_indices


def load_text_dataset_from_json(path):

    try:
        # Load the compressed JSON file
        with open(path, 'rb') as f:
            compressed_data = f.read()
    except:
        print('Invalid data path')

    # Decompress the data
    decompressed_data = zstd.decompress(compressed_data)

    # Decode the JSON content
    json_data = decompressed_data.decode('utf-8')

    # Parse the JSON data
    posts = json.loads(json_data)

    # Extract comments and GPT responses from each post
    data = []
    labels = []

    for post in posts:
        comments = [comment['body'] for comment in post['comments']]
        gpt_response = post['gpt']
        
        # Append comments and GPT response to the data list
        data.extend(comments)
        data.append(gpt_response)
        
        # Assign labels to indicate whether the content is chat GPT generated or not
        labels.extend([0] * len(comments))
        labels.append(1)

    # Shuffle the data and labels together
    data, labels = shuffle(data, labels)

    # # Print the shuffled data and labels
    # for d, label in zip(data, labels):
    #     print(f'Label: {label}, Content: {d}')

    # Count the occurrences of all terms in the dataset
    term_counts = Counter()

    # Initialize an empty dictionary to store term indices
    term_indices = {}

    for document in data:
        for term in re.findall(r"(\w+|[^\w\s])", document):
            if term not in term_indices:
                # Assign the next available index to the term
                term_indices[term] = len(term_indices)

    # Get the number of unique terms
    num_unique_terms = len(term_indices)

    return data, labels, num_unique_terms, term_indices


def build_text_tensor(window_size=5):

    option = input('Load from: \n\t (1) CSV\n\t (2) JSON\n')

    if option == '1':
        data, labels, total_num_terms, term_indices = load_text_dataset_from_csv(input('Enter CSV data path: '))
    elif option == '2':
        # Get data and labels from load_dataset()
        data, labels, total_num_terms, term_indices = load_text_dataset_from_json(input('Enter JSON data path: '))

    indices = []
    padded_slices = []

    # Loop over each document in the data
    for doc_idx, document in enumerate(data):

        terms = re.findall(r"(\w+|[^\w\s])", document)

        slice = np.zeros((total_num_terms, total_num_terms))
        # Loop over each of the terms
        for term_idx, term1 in enumerate(terms):

            # Loop over terms within window from i
            for term2 in terms[term_idx + 1: term_idx + window_size]:

                # Append indices of co-occurrence terms
                indices.append([doc_idx, term_indices[term1], term_indices[term2]])
                indices.append([doc_idx, term_indices[term2], term_indices[term1]])

                slice[term_indices[term1], term_indices[term2]] += 1
                slice[term_indices[term2], term_indices[term1]] += 1
            
        padded_slices.append(slice)

    i = torch.tensor(list(zip(*indices)))
    values = torch.ones(len(indices))

    tensor = sparse.COO(i, data=values)

    return tensor, padded_slices, labels, 'reddit'


def get_text_tensor_indices(window_size=5):

    option = input('Load from: \n\t (1) CSV\n\t (2) JSON\n')

    if option == '1':
        data, labels, total_num_terms, term_indices = load_text_dataset_from_csv(input('Enter CSV data path: '), subset_size=100)
    elif option == '2':
        # Get data and labels from load_dataset()
        data, labels, total_num_terms, term_indices = load_text_dataset_from_json(input('Enter JSON data path: '))

    indices = []
    tensor_size = (len(data), total_num_terms, total_num_terms)


    # Loop over each document in the data
    for doc_idx, document in enumerate(data):

        try:
            terms = re.findall(r"(\w+|[^\w\s])", document)
        except:
            print(f'Failed to process document #{doc_idx}')
            continue

        # Loop over each of the terms
        for term_idx, term1 in enumerate(terms):

            # Loop over terms within window from i
            for term2 in terms[term_idx + 1: term_idx + window_size]:

                # Append indices of co-occurrence terms
                indices.append([doc_idx, term_indices[term1], term_indices[term2]])
                indices.append([doc_idx, term_indices[term2], term_indices[term1]])

    return indices, tensor_size, 'reddit'

if __name__ == '__main__':

    # tensor, slices, labels = build_text_tensor()

    # print('Tensor Shape:', tensor.shape)
    # print('Slices Shape:', f'({len(slices)}, {slices[0].shape[0]}, {slices[0].shape[1]})')
    # print('Labels Shape:', len(labels))

    indices, tensor_size, dataset = get_text_tensor_indices()

    i = np.array(indices)
    values = np.ones(len(indices))

    # Create the directory if it doesn't exist
    if not os.path.exists('tensor_data/'):
        os.makedirs('tensor_data/')

    # Save indices and values to text files
    savemat(f'tensor_data/{dataset}_tensor_data.mat', {'indices': i, 'values':values, 'size':tensor_size})

