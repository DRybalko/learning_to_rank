import numpy as np
import random
import pdb
import math
import os, sys
import time

import gensim
from gensim.models.keyedvectors import KeyedVectors

def batch_gen(X, batch_size):
    # Borrowed from https://github.com/shashankg7/Keras-CNN-QA
   
    n_batches = X.shape[0]/float(batch_size)
    n_batches = int(math.ceil(n_batches))
    end = int(X.shape[0]/float(batch_size)) * batch_size
    n = 0
    for i in range(0,n_batches):
        if i < n_batches - 1: 
            if len(X.shape) > 1:
                batch = X[i*batch_size:(i+1) * batch_size, :]
                yield batch
            else:
                batch = X[i*batch_size:(i+1) * batch_size]
                yield batch
        
        else:
            if len(X.shape) > 1:
                batch = X[end: , :]
                n += X[end:, :].shape[0]
                yield batch
            else:
                batch = X[end:]
                n += X[end:].shape[0]
                yield batch

def load_embeddings(embedding_file, outdir, vocab):
    
    '''Load pre-learnt word embeddings.
    Return: embedding: embedding matrix with dim |vocab| x dim
            dim: dimension of the embeddings
            rand_count: number of words not in trained embedding
    '''
    print('Loading word vectors...')
    start = time.time()
    
    try:
        print('Trying to load from npy dump.')
        embedding = np.load(os.path.join(outdir, 'embedding.npy'))
        return embedding, embedding.shape[1], 'NA'
    except:
        print('Load from dump failed, reading from binary.')

    word_vectors = KeyedVectors.load_word2vec_format(
        embedding_file, binary=True)
    print('Loaded in %f seconds' %(time.time() - start))
    # Need to use the word vectors to make embeddings matrix
    # Get dimension for any word embedding
    dim = word_vectors['apple'].shape[0]
    
    # Initialize an embedding of |vocab| x dim
    # word -> embedding
    embedding = np.zeros((len(vocab), dim))
    # Take random values
    rand_vec = np.random.uniform(-0.25, 0.25, dim)
    # Count of words not having representations in our embedding file
    rand_count = 0

    for key, value in vocab.items():
        # Map word idx to its embedding vector.
        try:
            embedding[value] = word_vectors[key]
        except:
            embedding[value] = rand_vec
            rand_count += 1

    print('Total time for loading embedding: %f seconds' %(time.time() - start))
    print('Number of words not in trained embedding: %d' %(rand_count))

    np.save(os.path.join(outdir, 'embedding.npy'), embedding)
    return embedding, dim, rand_count
    