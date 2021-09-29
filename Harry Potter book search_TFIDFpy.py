from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import pandas as pd
import warnings
import numpy as np

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)


def read_data_file(filename):
    with open(filename, 'r') as fd:
        return fd.read()


def get_harry_potter():
    return read_data_file('hp1.txt')


def build_corpus(count=1):
    collection = ['hp1.txt', 'hp2.txt', 'hp3.txt', 'hp4.txt', 'hp5.txt', 'hp6.txt', 'hp7.txt']
    corpus = []
    for i in range(count):
        corpus.append(read_data_file(collection[i]))
    return corpus


def test_corpus():
    c = build_corpus(2)
    print(len(c) == 2)
    print(c[0][0:17])
    print(c[1][0:31])
    doc1 = c[0]
    print(len(doc1.split()), len(set(doc1.split())))


def load_stopwords(extra=[]):
    return extra + ['a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 'any', 'are',
                    "aren't", 'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but',
                    'by', "can't", 'cannot', 'could', "couldn't", 'did', "didn't", 'do', 'does', "doesn't", 'doing',
                    "don't", 'down', 'during', 'each', 'few', 'for', 'from', 'further', 'had', "hadn't", 'has',
                    "hasn't", 'have', "haven't", 'having', 'he', "he'd", "he'll", "he's", 'her', 'here', "here's",
                    'hers', 'herself', 'him', 'himself', 'his', 'how', "how's", 'i', "i'd", "i'll", "i'm", "i've", 'if',
                    'in', 'into', 'is', "isn't", 'it', "it's", 'its', 'itself', "let's", 'me', 'more', 'most',
                    "mustn't", 'my', 'myself', 'no', 'nor', 'not', 'of', 'off', 'on', 'once', 'only', 'or', 'other',
                    'ought', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 'same', "shan't", 'she', "she'd",
                    "she'll", "she's", 'should', "shouldn't", 'so', 'some', 'such', 'than', 'that', "that's", 'the',
                    'their', 'theirs', 'them', 'themselves', 'then', 'there', "there's", 'these', 'they', "they'd",
                    "they'll", "they're", "they've", 'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up',
                    'very', 'was', "wasn't", 'we', "we'd", "we'll", "we're", "we've", 'were', "weren't", 'what',
                    "what's", 'when', "when's", 'where', "where's", 'which', 'while', 'who', "who's", 'whom', 'why',
                    "why's", 'with', "won't", 'would', "wouldn't", 'you', "you'd", "you'll", "you're", "you've", 'your',
                    'yours', 'yourself', 'yourselves']


def build_tf_idf_model(docs, stopwords=[]):
    vectorizer = TfidfVectorizer(smooth_idf=True, use_idf=True, norm=None, sublinear_tf=True,
                                 stop_words=load_stopwords())
    corpus = docs
    matrix = vectorizer.fit_transform(corpus)
    return vectorizer, matrix


def test_build(corpus):
    vec, matrix = build_tf_idf_model(corpus)
    pass


def print_tfidf(vec, matrix, n=0):
    features = vec.get_feature_names()  # the unique words
    doc_vector = matrix[n]
    df = pd.DataFrame(doc_vector.T.todense(), index=features, columns=["tfidf"])
    df.sort_values(by=["tfidf"], ascending=False, inplace=True)
    print(df.head(10))


def test_print():
    corpus = build_corpus(1)
    vec, matrix = build_tf_idf_model(corpus)
    print_tfidf(vec, matrix)


# test_print()


def prepare_query_v1(corpus=None, query=''):
    if corpus is None:
        corpus = build_corpus(3)
    vec, matrix = build_tf_idf_model(corpus)
    transformed_query = vec.transform([query])
    return transformed_query


def dump_sparse_vector(v):
    coo_m = v.tocoo()
    for r, c, d in zip(coo_m.row, coo_m.col, coo_m.data):
        print('non zero at', r, c, d)
    return coo_m


def test_single_query(query):
    q_vec = prepare_query_v1(query=query)
    dump_sparse_vector(q_vec)


def print_matching_document(matrix, q_vector):
    assert q_vector.shape[0] == 1, "bad query vector (wrong size)"
    for m_idx, m in enumerate(matrix):
        for q_idx, q in enumerate(q_vector):
            print(cosine_similarity(m, q))


def find_matching_document(matrix, q_vector):
    print_matching_document(matrix, q_vector)
    results = cosine_similarity(matrix, q_vector).reshape((-1,))
    max_index = results.argsort()[-1]
    return max_index, results[max_index]


def find_match(query='', corpus_size=3):
    corpus = build_corpus(corpus_size)
    vec, matrix = build_tf_idf_model(corpus)
    q_vector = prepare_query_v1(corpus, query)
    return find_matching_document(matrix, q_vector)


def show_image(path):
    with open(path, 'rb') as fd:
        fig, ax = plt.subplots()
        ax.set_axis_off()
        imgdata = plt.imread(fd)
        im = ax.imshow(imgdata)
        return fig


def test_UI(vec=None, matrix=None, query='', debug=False):
    labels = ['hp1.png', 'hp2.png', 'hp3.png', 'hp4.png', 'hp5.png', 'hp6.png', 'hp7.png']
    if vec is None or matrix is None:
        corpus = build_corpus(7)
        corpus_size = 7
        vec, matrix = build_tf_idf_model(corpus)

    corpus_size = input("Enter the number of volumes you want to search out of total 7 volumes: ")
    idx, cosmax = find_match(query, int(corpus_size))
    path = labels[idx]
    show_image(path)
    plt.show()
    return idx

