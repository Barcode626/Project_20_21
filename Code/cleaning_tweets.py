#!/usr/bin/env python
# coding: utf-8

import re
import pandas as pd
import numpy as np


def clean_hashtags(hg):
    if pd.isna(hg):
        return hg     
    else:
        hg = eval(hg)
        j = []
        # Remove all the non-English hashtags
        for i in hg:
            i = re.sub(r'[^a-zA-Z]', '', i)
            if i != '':
                j.append(i)
        j = str(j)
        j = re.sub("'", '', j)
        j = re.sub("\[", '', j)
        j = re.sub("\]", '', j)
        return j


def clean_geography(gg, k):
    if pd.isna(gg)==True:
        return np.nan
    else:
        if k in list(eval(gg).keys()):
            return eval(gg)[k]
        else:
            return np.nan


def clean_tweets(content):
    # Remove usernames mentioned
    content = re.sub('@[^\s]+', ' ', content)
    
    # Remove URLs
    content = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', ' ', content)
    
    # Remove extra white space and alphabet
    content = re.sub('&amp;', '&', content)
    content = re.sub(' +', ' ', content)
    content = content.strip()
    
    # Convert to lowercase
    content = content.lower()
    
    return content

def clean_aspects(ap):
    if ap == '[]' or pd.isna(ap) == True:
        return np.nan
    else:
        return ap



# LDA
# Gensim
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
import gensim.corpora as corpora
from gensim.models import CoherenceModel

# NLTK
import nltk
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer 
from nltk.stem.porter import * 

def preprocess(content):
    results = []
    for token in gensim.utils.simple_preprocess(content):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            results.append(WordNetLemmatizer().lemmatize(token))
    return results

model_path = "/content/drive/MyDrive/Code/Model/"
lda_model = gensim.models.LdaModel.load(model_path + "LDA_Model")
dictionary = gensim.corpora.Dictionary.load(model_path + "dictionary.dict")
bow_corpus = corpora.MmCorpus(model_path + "bow_corpus.mm")

topicidx_to_aspect = {'0': 'lockdown', '1': 'government', '2': 'lockdown', \
                                '3': 'protective measures', '4': 'china', '8': 'lockdown', \
                                '10': 'lockdown', '11': 'treatment', '12': 'support measures', \
                                '13': 'protective measures', '15': 'lockdown', '16': 'support measures', \
                                '18': 'support measures', '20': 'quarantine', '21': 'spread', \
                                '23': 'information', '25': 'support measures', '26': 'lockdown', \
                                '27': 'spread', '28': 'protective measures', '29': 'government'}
df_aspect_idx = pd.read_excel("/content/drive/MyDrive/Data/topic/aspect2idx.xlsx")
df_discard = pd.read_excel("/content/drive/MyDrive/Data/topic/discard.xlsx")
aspect_to_idx = {}
for idx, ap in enumerate(list(df_aspect_idx['aspects'])):
    aspect_to_idx[ap] = idx

idx_to_aspect = {}
for ap, idx in aspect_to_idx.items():
    idx_to_aspect[str(idx)] = ap

def getidx():
    return topicidx_to_aspect, aspect_to_idx, idx_to_aspect

def dealing_topics(content):
    lda_vector = lda_model[dictionary.doc2bow(content)]
    topics = []
    for idx in range(len(lda_vector)):
        if lda_vector[idx][0] in list(df_discard['topic_index']):
            continue
        else:
            topics.append(lda_vector[idx])
    topics_sorted = sorted(topics, key=lambda tup: -1*tup[1])
    return topics_sorted

def generate_aspects(content):
    aspect_scores = {}
    for topic_idx, scores in content:
        aspect = topicidx_to_aspect[str(topic_idx)]
        if aspect in aspect_scores.keys():
            aspect_scores[aspect] += scores
        else:
            aspect_scores[aspect] = scores
    aspects_sorted = sorted(aspect_scores.items(), key = lambda x: x[1], reverse = True)
    aspects_sorted = [(aspect_to_idx[aspect], score) for aspect, score in aspects_sorted if score>=0.1][:3]
    return aspects_sorted



# wordcloud hashtags
def preprocess_hg(content):
    results = []
    for token in gensim.utils.simple_preprocess(content):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 1:
            results.append(WordNetLemmatizer().lemmatize(token))
    return results


def dva(content, idx):
    if pd.isna(content):
        return np.nan, np.nan
    else:
        lst = list(eval(content))
        ap_idx_lst = [int(apidx) for apidx, score in lst]
        ap_score_lst = [float(score) for apidx, score in lst]
        if int(idx) in ap_idx_lst:
            return str(idx_to_aspect[str(idx)]), ap_score_lst[ap_idx_lst.index(int(idx))]
        else:
            return np.nan, np.nan