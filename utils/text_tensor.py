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

    for document in data:
        for term in document.split():
            if term not in term_indices:
                # Assign the next available index to the term
                term_indices[term] = len(term_indices)

    # Get the number of unique terms
    num_unique_terms = len(term_indices)

    return data, labels, num_unique_terms, term_indices


def build_text_tensor(window_size=5):

    # Get data and labels from load_dataset()
    data, labels, total_num_terms, term_indices = load_dataset(input('Enter JSON data path: '))

    indices = []
    padded_slices = []

    # Loop over each document in the data
    for doc_idx, document in enumerate(data):

        terms = document.split()

       # Loop over each of the terms
        for term_idx, term1 in enumerate(terms):

            slice = np.zeros((total_num_terms, total_num_terms))
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

if __name__ == '__main__':

    tensor, slices, labels = build_text_tensor()

    print('Tensor Shape:', tensor.shape)
    print('Slices Shape:', f'({len(slices)}, {slices[0].shape[0]}, {slices[0].shape[1]})')
    print('Labels Shape:', len(labels))