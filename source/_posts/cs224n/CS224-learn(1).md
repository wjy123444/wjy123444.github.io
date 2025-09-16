---
title: CS224-learn(1)
date: 2025-03-30 09:30:42
tags: CS224
categories: 学习
---
记录一下自己做CS224作业的过程和一些细节

顺带一提这种作业形式真挺先进的，老师为你提供大部分内容，你只需要填入关键内容，一定程度上可能会让你偷点懒吧，但是能节省大部分的时间，而且能使作业的要求十分明确，不至于学生不知所云

# Import

```python
import sys
assert sys.version_info[0]==3
assert sys.version_info[1] >= 5

from gensim.models import KeyedVectors
from gensim.test.utils import datapath
import pprint
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [10, 5]
import nltk
nltk.download('reuters')
from nltk.corpus import reuters
import numpy as np
import random
import scipy as sp
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import PCA

START_TOKEN = '<START>'
END_TOKEN = '<END>'

np.random.seed(0)
random.seed(0)
```

# Read_corpus

```python
def read_corpus(category="crude"):
    """ Read files from the specified Reuter's category.
        Params:
            category (string): category name
        Return:
            list of lists, with words from each of the processed files
    """
    files = reuters.fileids(category)
    return [[START_TOKEN] + [w.lower() for w in list(reuters.words(f))] + [END_TOKEN] for f in files]

reuters_corpus = read_corpus()
pprint.pprint(reuters_corpus[:3], compact=True, width=100)
```

这里 retuers.fileids(category) 会返回一个列表，其中每一个元素使一个字符串，字符串的值是Reuter 数据集中某个字符串的ID，类似于‘test/12345’

retuers.words(f)用于返回对应的ID的所有单词，注意python中列表推导式的运用

# Distinct_words

```python
def distinct_words(corpus):
    """ Determine a list of distinct words for the corpus.
        Params:
            corpus (list of list of strings): corpus of documents
        Return:
            corpus_words (list of strings): sorted list of distinct words across the corpus
            num_corpus_words (integer): number of distinct words across the corpus
    """
    corpus_words = []
    num_corpus_words = -1
  
    # ------------------
    # Write your implementation here.
    corpus_words = [word for doc in corpus for word in doc]
    corpus_words = sorted(list(set(corpus_words)))
    num_corpus_words = len(corpus_words)

    # ------------------

    return corpus_words, num_corpus_words
```

## test

```python
test_corpus = ["{} All that glitters isn't gold {}".format(START_TOKEN, END_TOKEN).split(" "), "{} All's well that ends well {}".format(START_TOKEN, END_TOKEN).split(" ")]
test_corpus_words, num_corpus_words = distinct_words(test_corpus)

# Correct answers
ans_test_corpus_words = sorted([START_TOKEN, "All", "ends", "that", "gold", "All's", "glitters", "isn't", "well", END_TOKEN])
ans_num_corpus_words = len(ans_test_corpus_words)

# Test correct number of words
assert(num_corpus_words == ans_num_corpus_words), "Incorrect number of distinct words. Correct: {}. Yours: {}".format(ans_num_corpus_words, num_corpus_words)

# Test correct words
assert (test_corpus_words == ans_test_corpus_words), "Incorrect corpus_words.\nCorrect: {}\nYours:   {}".format(str(ans_test_corpus_words), str(test_corpus_words))

# Print Success
print ("-" * 80)
print("Passed All Tests!")
print ("-" * 80)
```

# Compute_co_occurrence_matrix

这里要求以窗口大小为4实现共现矩阵的构建

```python
def compute_co_occurrence_matrix(corpus, window_size=4):
    """ Compute co-occurrence matrix for the given corpus and window_size (default of 4).
  
        Note: Each word in a document should be at the center of a window. Words near edges will have a smaller
              number of co-occurring words.
        
              For example, if we take the document "<START> All that glitters is not gold <END>" with window size of 4,
              "All" will co-occur with "<START>", "that", "glitters", "is", and "not".
  
        Params:
            corpus (list of list of strings): corpus of documents
            window_size (int): size of context window
        Return:
            M (a symmetric numpy matrix of shape (number of unique words in the corpus , number of unique words in the corpus)): 
                Co-occurence matrix of word counts. 
                The ordering of the words in the rows/columns should be the same as the ordering of the words given by the distinct_words function.
            word2ind (dict): dictionary that maps word to index (i.e. row/column number) for matrix M.
    """
    words, num_words = distinct_words(corpus)
    M = None
    word2ind = {}
    word2ind = {word: i for i, word in enumerate(words)}
    # ------------------
    # Write your implementation here.
    M = np.zeros((num_words, num_words))
    for body in corpus:
        for curr_idx, word in enumerate(body):
            for window_idx in range(-window_size, window_size + 1):
                neighbor_idx = curr_idx + window_idx
                if (neighbor_idx < 0) or (neighbor_idx >= len(body)) or (curr_idx == neighbor_idx):
                    continue
                co_occur_word = body[neighbor_idx]
                (word_idx, co_occur_idx) = (word2ind[word],word2ind[co_occur_word])
                M[word_idx, co_occur_idx] += 1

    # ------------------

    return M, word2ind
```

构建一个映射到共现矩阵的字典，再一句话一句话的搜索

## test

```python
test_corpus = ["{} All that glitters isn't gold {}".format(START_TOKEN, END_TOKEN).split(" "), "{} All's well that ends well {}".format(START_TOKEN, END_TOKEN).split(" ")]
M_test, word2ind_test = compute_co_occurrence_matrix(test_corpus, window_size=1)

# Correct M and word2ind
M_test_ans = np.array( 
    [[0., 0., 0., 0., 0., 0., 1., 0., 0., 1.,],
     [0., 0., 1., 1., 0., 0., 0., 0., 0., 0.,],
     [0., 1., 0., 0., 0., 0., 0., 0., 1., 0.,],
     [0., 1., 0., 0., 0., 0., 0., 0., 0., 1.,],
     [0., 0., 0., 0., 0., 0., 0., 0., 1., 1.,],
     [0., 0., 0., 0., 0., 0., 0., 1., 1., 0.,],
     [1., 0., 0., 0., 0., 0., 0., 1., 0., 0.,],
     [0., 0., 0., 0., 0., 1., 1., 0., 0., 0.,],
     [0., 0., 1., 0., 1., 1., 0., 0., 0., 1.,],
     [1., 0., 0., 1., 1., 0., 0., 0., 1., 0.,]]
)
ans_test_corpus_words = sorted([START_TOKEN, "All", "ends", "that", "gold", "All's", "glitters", "isn't", "well", END_TOKEN])
word2ind_ans = dict(zip(ans_test_corpus_words, range(len(ans_test_corpus_words))))

# Test correct word2ind
assert (word2ind_ans == word2ind_test), "Your word2ind is incorrect:\nCorrect: {}\nYours: {}".format(word2ind_ans, word2ind_test)

# Test correct M shape
assert (M_test.shape == M_test_ans.shape), "M matrix has incorrect shape.\nCorrect: {}\nYours: {}".format(M_test.shape, M_test_ans.shape)

# Test correct M values
for w1 in word2ind_ans.keys():
    idx1 = word2ind_ans[w1]
    for w2 in word2ind_ans.keys():
        idx2 = word2ind_ans[w2]
        student = M_test[idx1, idx2]
        correct = M_test_ans[idx1, idx2]
        if student != correct:
            print("Correct M:")
            print(M_test_ans)
            print("Your M: ")
            print(M_test)
            raise AssertionError("Incorrect count at index ({}, {})=({}, {}) in matrix M. Yours has {} but should have {}.".format(idx1, idx2, w1, w2, student, correct))

# Print Success
print ("-" * 80)
print("Passed All Tests!")
print ("-" * 80)
```

# Reduce_to_k_dim

构造一个共现矩阵会遇到维度太高的问题，使用SVD方法来截断奇异值，使词向量在保持良好特性的同时降低维度

```python
def reduce_to_k_dim(M, k=2):
    """ Reduce a co-occurence count matrix of dimensionality (num_corpus_words, num_corpus_words)
        to a matrix of dimensionality (num_corpus_words, k) using the following SVD function from Scikit-Learn:
            - http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html
  
        Params:
            M (numpy matrix of shape (number of unique words in the corpus , number of unique words in the corpus)): co-occurence matrix of word counts
            k (int): embedding size of each word after dimension reduction
        Return:
            M_reduced (numpy matrix of shape (number of corpus words, k)): matrix of k-dimensioal word embeddings.
                    In terms of the SVD from math class, this actually returns U * S
    """  
    n_iters = 10     # Use this parameter in your call to `TruncatedSVD`
    M_reduced = None
    print("Running Truncated SVD over %i words..." % (M.shape[0]))
  
        # ------------------
        # Write your implementation here.
    svd = TruncatedSVD(n_components = k)
    svd.fit(M)
    M_reduced = svd.transform(M)
  
        # ------------------

    print("Done.")
    return M_reduced
```

1. TruncatedSVD(n_components= k)

功能: 这是scikit-learn库中的一个类，用于执行截断奇异值分解（Truncated Singular Value Decomposition, SVD），通常用于降维。

参数:  n_components: 指定要保留的奇异值数量（即降维后的维度数）。在这个例子中，k表示目标维度。

返回值:  返回一个TruncatedSVD对象，该对象包含降维所需的信息（如奇异值、右奇异向量等）。

2. svd.fit(M)

功能: 使用矩阵M来拟合TruncatedSVD模型，计算奇异值分解所需的参数。

参数:  M: 输入矩阵，形状为(n_samples, n_features)。在这里，M是一个共现矩阵，表示单词之间的共现关系。

返回值:  无返回值（None），但会更新svd对象内部的状态，例如保存奇异值和右奇异向量。

3. svd.transform(M)

功能: 将输入矩阵M投影到由fit方法计算出的低维空间中。

参数:  M: 输入矩阵，与fit方法中的矩阵相同或具有相同的特征空间。

返回值:  返回一个形状为(n_samples, k)的降维矩阵M_reduced，其中k是由n_components指定的目标维度。这个矩阵表示原始数据在低维空间中的表示。

## test

```python
test_corpus = ["{} All that glitters isn't gold {}".format(START_TOKEN, END_TOKEN).split(" "), "{} All's well that ends well {}".format(START_TOKEN, END_TOKEN).split(" ")]
M_test, word2ind_test = compute_co_occurrence_matrix(test_corpus, window_size=1)
M_test_reduced = reduce_to_k_dim(M_test, k=2)

# Test proper dimensions
assert (M_test_reduced.shape[0] == 10), "M_reduced has {} rows; should have {}".format(M_test_reduced.shape[0], 10)
assert (M_test_reduced.shape[1] == 2), "M_reduced has {} columns; should have {}".format(M_test_reduced.shape[1], 2)

# Print Success
print ("-" * 80)
print("Passed All Tests!")
print ("-" * 80)
```

# GLoVe

```python
def load_embedding_model():
    """ Load GloVe Vectors
        Return:
            wv_from_bin: All 400000 embeddings, each lengh 200
    """
    import gensim.downloader as api
    wv_from_bin = api.load("glove-wiki-gigaword-200")
    print("Loaded vocab size %i" % len(wv_from_bin.key_to_index.keys()))
    return wv_from_bin

wv_from_bin = load_embedding_model()
```

# Get_GloVe_matrix and reduce to k-dimension

```python
def get_matrix_of_vectors(wv_from_bin, required_words=['barrels', 'bpd', 'ecuador', 'energy', 'industry', 'kuwait', 'oil', 'output', 'petroleum', 'iraq']):
    """ Put the GloVe vectors into a matrix M.
        Param:
            wv_from_bin: KeyedVectors object; the 400000 GloVe vectors loaded from file
        Return:
            M: numpy matrix shape (num words, 200) containing the vectors
            word2ind: dictionary mapping each word to its row number in M
    """
    import random
    words = list(wv_from_bin.key_to_index.keys())
    print("Shuffling words ...")
    random.seed(224)
    random.shuffle(words)
    words = words[:10000]
    print("Putting %i words into word2ind and matrix M..." % len(words))
    word2ind = {}
    M = []
    curInd = 0
    for w in words:
        try:
            M.append(wv_from_bin.word_vec(w))
            word2ind[w] = curInd
            curInd += 1
        except KeyError:
            continue
    for w in required_words:
        if w in words:
            continue
        try:
            M.append(wv_from_bin.word_vec(w))
            word2ind[w] = curInd
            curInd += 1
        except KeyError:
            continue
    M = np.stack(M)
    print("Done.")
    return M, word2ind
M, word2ind = get_matrix_of_vectors(wv_from_bin)
M_reduced = reduce_to_k_dim(M, k=2)

```

# Similarity and thinking

```python
w1 = wv_from_bin['good']
w2 = wv_from_bin['bad']
w3 = wv_from_bin['great']

w1_w2_dist = wv_from_bin.distance('good', 'bad')
w1_w3_dist = wv_from_bin.distance('good', 'great')
print(w1_w2_dist, w1_w3_dist)

pprint.pprint(wv_from_bin.most_similar(positive = ['woman', 'king'],negative = ['man']))
pprint.pprint(wv_from_bin.most_similar(positive = ['hand', 'sock'],negative = ['foot']))
```
