import json
import random

import json
import zstandard as zstd

from collections import Counter

import torch
import numpy as np

from sklearn.utils import shuffle
import sparse


def load_dataset(path):

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

    for item in data:
        # terms = item.split()
        # term_counts.update(terms)
        for term in item:
            if term not in term_indices:
                # Assign the next available index to the term
                term_indices[term] = len(term_indices)

    # Get the number of unique terms
    num_unique_terms = len(term_indices)

    return data, labels, num_unique_terms, term_indices


def build_text_tensor(window_size=5):

    # Get data and labels from load_dataset()
    data, labels, N, term_indices = load_dataset(input('Enter JSON data path: '))

    indices = []
    padded_slices = []

    # Loop over each item in the data
    for i, item in enumerate(data):

        terms = item.split()

       # Loop over each of the terms
        for i, term1 in enumerate(terms):

            slice = np.zeros((N, N))
            # Loop over terms within window from i
            for j, term2 in enumerate(terms[i + 1: i + window_size]):

                # Append indices of co-occurrence terms
                indices.append([i, term_indices[term1], term_indices[term2]])
                indices.append([i, term_indices[term2], term_indices[term1]])

                slice[term_indices[term1], term_indices[term2]] += 1
                slice[term_indices[term2], term_indices[term1]] += 1
            
            padded_slices.apend(slice)

    i = torch.tensor(list(zip(*indices)))
    values = torch.ones(len(indices))

    tensor = sparse.COO(i, data=values)

    return tensor, padded_slices, labels
