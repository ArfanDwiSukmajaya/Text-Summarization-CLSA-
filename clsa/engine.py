import pandas as pd
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory 
import string
from nltk.stem import PorterStemmer
import re
from sklearn.feature_extraction.text import TfidfVectorizer # tf-idf
import scipy as sp
import numpy as np
from scipy.linalg import svd 
from numpy import dot
from nltk.stem import PorterStemmer # preprocessing bahasa inggris
from scipy.linalg import svd
import numpy

stemmer = StemmerFactory().create_stemmer()  # Object stemmer
remover = StopWordRemoverFactory().create_stop_word_remover()  # objek stopword
translator = str.maketrans('', '', string.punctuation)


def stemmerEN(text):
    porter = PorterStemmer()
    ## stop = set(stopwords.words('english')) #stopwods berguna untuk
    factory = StopWordRemoverFactory()
    stopwords = factory.get_stop_words()
    text = text.lower()
    text = [i for i in text.lower().split() if i not in stopwords]
    text = ' '.join(text)
    preprocessed_text = text.translate(translator)
    text_stem = porter.stem(preprocessed_text)
    return text_stem

def preprocessing(text):
    text = text.lower()
    text_clean = remover.remove(text)
    text_stem = stemmer.stem(text_clean)
    text_stem = stemmerEN(text_stem)
    return text_stem

def splitParagraphIntoSentences(paragraph):
    sentenceEnders = re.compile('[.!?]')
    sentenceList = sentenceEnders.split(paragraph)
    return sentenceList

def TFIDF(doc) :  
    factory = StopWordRemoverFactory()
    stopwords = factory.get_stop_words()

    tfidf_vectorizer = TfidfVectorizer(stop_words=stopwords,smooth_idf=False, norm=None)
    X = tfidf_vectorizer.fit_transform(doc)
    pd.DataFrame(X.toarray(), columns=tfidf_vectorizer.get_feature_names())
    return X
    #return pd.DataFrame(X.toarray(), columns=tfidf_vectorizer.get_feature_names())

def LSA(tfidf): 
    matrixAT=tfidf.toarray()
    matrixA=np.transpose(matrixAT)
    matrixAT=np.transpose(matrixA)
    a = np.array(matrixAT)
    b = np.array(matrixA)
    aTa = dot(a,b)
    tempSVD = np.array(aTa)
    u, s, v = np.linalg.svd(tempSVD, full_matrices = False)
    s = sp.diag(s)
    loop = s[0]
    for i in np.arange(np.size(loop)):
        for j in np.arange(np.size(loop)):
            s[i][j] = np.sqrt(dot(np.square(s[i][j]),np.square(v[i][j])))
    return s

def CLSA(tfidf):
    matrixAT=tfidf.toarray()
    matrixA=numpy.transpose(matrixAT)
    matrixAT=numpy.transpose(matrixA)
    a = np.array(matrixAT)
    b = np.array(matrixA)
    aTa = dot(a,b)
    A = np.array(aTa)
    u, s, v = np.linalg.svd(A, full_matrices = False)
    s = sp.diag(s)
    aa =v[0]
    for i in np.arange(np.size(aa)):
        av = np.average(v[i])
        for j in np.arange(np.size(aa)):
            if v[i][j] < av :
                v[i][j] = 0
    loop = s[0]
    for i in np.arange(np.size(loop)):
        for j in np.arange(np.size(loop)):
            s[i][j] = np.sqrt(dot(np.square(s[i][j]),np.square(v[i][j])))
    return s

def summary_sentence(doc, result_TFIDF, params_angka, types=''):
    types = types.lower()
    # try :
    if (types == 'lsa') :
        matrixAT=result_TFIDF.toarray()
        matrixA=np.transpose(matrixAT)
        matrixAT=np.transpose(matrixA)
        a = np.array(matrixAT)
        b = np.array(matrixA)
        aTa = dot(a,b)
        tempSVD = np.array(aTa)
        u, s, v = svd(tempSVD)
        s = sp.diag(s)
        datas = list()
        maks = int(params_angka)  
        
        loop = s[0]
        temp = 0
        temp1 = []
        for i in np.arange(np.size(loop)):
            for j in np.arange(np.size(loop)):
                temp = temp + np.sqrt(dot(np.square(s[i][j]),np.square(v[i][j])))
            temp1.append(temp)
            temp = 0
        #temp2=[]
        #for i in range(0, len(temp1)):
        #    for j in range(i+1,len(temp1)):
        #        if(temp1[i]<temp1[j]):
        #            temp2=temp1[i]
        #            temp1[i]=temp1[j]
        #            temp1[j]=temp2
        #temp3 = []
        #if(maks <= len(loop)):
        #    for k in range(0,maks):
        #        temp4 = 0
        #        for i in np.arange(np.size(loop)):
        #            for j in np.arange(np.size(loop)):
        #                temp4 = temp4 + np.sqrt(dot(np.square(s[i][j]),np.square(v[i][j])))
        #            if(temp4 == temp1[k]):
        #                if(temp4 != 0):
        #                    temp3= temp3 + [[i, doc[i]]]
        #            temp4 = 0
        #temp5=[]
        #for i in range(0, len(temp3)):
        #    for j in range(i+1,len(temp3)):
        #        if(temp3[i]>temp3[j]):
        #            temp5=temp3[i]
        #            temp3[i]=temp3[j]
        #            temp3[j]=temp5
        #for i in range(0, len(temp3)):
        #    datas.append(temp3[i][1]+".")
        
        datas = max_summ(temp1, maks, loop, s, v, doc)
        
        return datas
            
    elif (types == 'clsa') :
        matrixAT=result_TFIDF.toarray()
        matrixA=np.transpose(matrixAT)
        matrixAT=np.transpose(matrixA)
        a = np.array(matrixAT)
        b = np.array(matrixA)
        aTa = dot(a,b)
        tempSVD = np.array(aTa)
        u, s, v = svd(tempSVD)
        s = sp.diag(s)
        aa =v[0]
        data = list()
        maks = int(params_angka) 


        for i in np.arange(np.size(aa)):
            av = np.average(v[i])
            for j in np.arange(np.size(aa)):
                if v[i][j] < av :
                    v[i][j] = 0

        loop = s[0]
        temp = 0
        temp1 = []
        for i in np.arange(np.size(loop)):
            for j in np.arange(np.size(loop)):
                temp = temp + np.sqrt(dot(np.square(s[i][j]),np.square(v[i][j])))
            temp1.append(temp)
            temp = 0
        data = max_summ(temp1, maks, loop, s, v, doc)
        return data
        
def sum_frame_by_column(frame, new_col_name, list_of_cols_to_sum):
    frame[new_col_name] = frame[list_of_cols_to_sum].astype(float).sum(axis=1)
    return(frame)

def max_summ(temp1, maks, loop, s, v, doc):
    this = list()
    temp2=[]
    for i in range(0, len(temp1)):
        for j in range(i+1,len(temp1)):
            if(temp1[i]<temp1[j]):
                temp2=temp1[i]
                temp1[i]=temp1[j]
                temp1[j]=temp2
    temp3 = []
    if(maks <= len(loop)):
        for k in range(0,maks):
            temp4 = 0
            for i in np.arange(np.size(loop)):
                for j in np.arange(np.size(loop)):
                    temp4 = temp4 + np.sqrt(dot(np.square(s[i][j]),np.square(v[i][j])))
                if(temp4 == temp1[k]):
                    if(temp4 != 0):
                        temp3= temp3 + [[i, doc[i]]]
                temp4 = 0
    temp5=[]
    for i in range(0, len(temp3)):
        for j in range(i+1,len(temp3)):
            if(temp3[i]>temp3[j]):
                temp5=temp3[i]
                temp3[i]=temp3[j]
                temp3[j]=temp5
    for i in range(0, len(temp3)):
            this.append(temp3[i][1]+".")
    return this

