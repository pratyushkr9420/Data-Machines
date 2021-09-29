import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer


def get_corpus():
    c1 = "Do you like Green eggs and ham"
    c2 = "I do not like them Sam I am I do not like Green eggs and ham"
    c3 = "Would you like them Here or there"
    c4 = "I would not like them Here or there I would not like them Anywhere"
    return [c1, c2, c3, c4]


def cv_demo1():
    corpus = get_corpus()
    cvec = CountVectorizer(lowercase=True)
    doc_term_matrix = cvec.fit_transform(corpus)
    print(cvec.get_feature_names())
    print(doc_term_matrix.toarray())


def split_into_tokens(data, normalize=True, min_length=0):
    word_list = []

    if normalize is True:
        data = data.lower()

    data = data.split()

    for i in data:
        if len(data) > min_length:
            word_list.append(i)

    return word_list


def cv_demo2():
    corpus = get_corpus()
    cvec = CountVectorizer(tokenizer=split_into_tokens)
    doc_term_matrix = cvec.fit_transform(corpus)
    tokens = cvec.get_feature_names()

    return doc_term_matrix, tokens


def word_matrix_to_df(wm, feature_names):
    doc_names = ['Doc{:d}'.format(idx + 1) for idx, _ in enumerate(wm)]
    df = pd.DataFrame(data=wm.toarray(), index=doc_names, columns=feature_names)
    return df


def cv_demo3():
    doc_term_matrix, tokens = cv_demo2()
    df = word_matrix_to_df(doc_term_matrix, tokens)
    return df


def cv_demo_idf():
    doc_term_matrix, tokens = cv_demo2()
    tfidf_transformer = TfidfTransformer()
    tfidf_transformer.fit(doc_term_matrix)
    df = pd.DataFrame(tfidf_transformer.idf_, index=tokens, columns=["idf_weights"])
    df.sort_values(by=['idf_weights'], inplace=True, ascending=False)
    return df


def cv_demo_tf_idf():
    doc_term_matrix, tokens = cv_demo2()
    tfidf_transformer = TfidfTransformer(smooth_idf=True)
    tfidf_transformer.fit(doc_term_matrix)
    idf = tfidf_transformer.idf_
    tf_idf_vector = tfidf_transformer.transform(doc_term_matrix)
    print(tf_idf_vector)


def cv_demo_pd_tf_idf():
    doc_term_matrix, tokens = cv_demo2()
    tfidf_transformer = TfidfTransformer(smooth_idf=True, sublinear_tf=True, norm=None)
    tfidf_transformer.fit(doc_term_matrix)
    idf = tfidf_transformer.idf_
    tf_idf_vector = tfidf_transformer.transform(doc_term_matrix)
    token = 'i'
    doc = 1
    df_idf = pd.DataFrame(idf, index=tokens, columns=["idf_weights"])
    df_idf.sort_values(by=['idf_weights'], inplace=True, ascending=False)
    idf_token = df_idf.loc[token]['idf_weights']
    doc_vector = tf_idf_vector[doc]
    df_tfidf = pd.DataFrame(doc_vector.T.todense(), index=tokens, columns=["tfidf"])
    df_tfidf.sort_values(by=["tfidf"], ascending=False, inplace=True)
    tfidf_token = df_tfidf.loc[token]['tfidf']
    tf_token = tfidf_token / idf_token
    print('TF {:s} {:2.4f}'.format(token, tf_token))
    print('IDF {:s} {:2.4f}'.format(token, idf_token))
    print('TFIDF {:s} {:2.4f}'.format(token, tfidf_token))


def dump_sparse_matrix():
    vec = TfidfVectorizer(use_idf=True)
    corpus = ["another day of rain; rain rain go away, comeback another day"]
    matrix = vec.fit_transform(corpus)
    print(matrix.shape)
    print(vec.idf_)
    coo_format = matrix.tocoo()
    print(coo_format.col)
    print(coo_format.data)
    tuples = zip(coo_format.col, coo_format.data)
    in_order = sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)
    features = vec.get_feature_names() # the unique words
    print(features)
    for score in in_order:
        idx = score[0]
        word = features[idx]
        print("{:10s} tfidf:".format(word), score)


'''
***********************************************
Review Questions
Q1)What does CountVectorizer do?
Q2)What does TfidfTransformer do?
Q3)What does sklearn's fit function do?
Q4)What does sklearn's transform function do?

## It converts a text to a vector on the basis of the frequency of each word that occurs in the text.
## It converts the vector of the word count to a TF.IDF matrix. 
## Sklearn's fit attempts to build & train the model based on the data provided. In this case based on the given documents fit builds idf vector.
## Sklearn's transform applies what was fitted using fit to incoming data. In this case it was used to create the TF.IDF vector
***********************************************
'''