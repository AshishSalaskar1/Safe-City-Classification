import gdown
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import nltk
import string
import re
import os
import time
import pickle

from xgboost import XGBClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from textblob import TextBlob


from collections import Counter
import textstat
from nltk.corpus import stopwords
from nltk import pos_tag, word_tokenize, RegexpParser
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score, f1_score, precision_score, recall_score
# import gdown

nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download("stopwords")
nltk.download('punkt')

class_label_names = {0:"None",1:"Staring",2:"Groping",3:"Commenting",4:"Commenting & Staring",5:"Staring & Groping",6:"Commenting & Groping",7:"Commenting, Staring and Groping"}
class_labels = {0:[0,0,0], 1:[0,1,0], 2:[0,0,1], 3:[1,0,0], 4:[1,1,0], 5:[0,1,1], 6:[1,0,1], 7:[1,1,1]}
multi_label_to_class = {'[0 0 0]':0,'[0 1 0]':1,'[0 0 1]':2,'[1 0 0]':3,'[0 0 0]':4,'[0 0 0]':5,'[0 0 0]':6,'[0 0 0]':7}

# def download_drive(drive_id, output):
#     url = "https://drive.google.com/uc?id="+drive_id
#     output = output

#     if output not in set(os.listdir(".")):
#       gdown.download(url, output, quiet=False)


# download_drive("1Z6bjXmyCaoEzXYo_tRDwLTsfeA2F3K3j","glove_vectors")

with open('glove.pickle', 'rb') as f:
  glove_vectors = pickle.load(f)
  glove_words = set(glove_vectors.keys())


def pre_process_text(phrase):
    '''Clean and lemmatize text'''
    phrase = re.sub(r"won't", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    phrase = re.sub('[^A-Za-z0-9]+', ' ', phrase)
    phrase = ' '.join(e.lower() for e in phrase.split() if e.lower() not in stopwords.words('english'))
    cleanr = re.compile('<.*?>')
    phrase = re.sub(cleanr, ' ', str(phrase))
    phrase = re.sub(r'[?|!|\'|"|#]',r'',phrase)
    phrase = re.sub(r'[.|,|)|(|\|/]',r' ',phrase)
    phrase = phrase.strip()
    phrase = phrase.replace("\n"," ")
    phrase = TextBlob(phrase).correct()
    lemmatizer = WordNetLemmatizer()
    lemmatized_sentence = " ".join([lemmatizer.lemmatize(word) for word in phrase.split(" ")])
    return str(lemmatized_sentence)

def count_POS(text, part_of_speech):
  ''' Count POS tokens'''
  pos_re = {"noun":re.compile('N.*'), "verb":re.compile('V.*'),
          "adverb":re.compile('R.*'), "adjective":re.compile('J.*'),
          "pronoun":re.compile('J.*')}
  count = 0
  pos_tokens = nltk.pos_tag(word_tokenize(text))
  for word, pos in pos_tokens:
    if re.match(pos_re[part_of_speech],pos):
      count += 1
  return count


# TF-IDF WEIGHTED W2V
def tfdidf_w2v(data, tf_idf_words, tfidf_dict):
  '''
   Find tfidf-weighted Word2Veca
  '''
  tfidf_w2v_vectors = []; 
  for sentence in data:
      vector = np.zeros(300) 
      tf_idf_weight =0
      for word in sentence.split(): 
          if (word in glove_words) and (word in tf_idf_words):
              vec = glove_vectors[word] 
              tf_idf = tfidf_dict[word]*(sentence.count(word)/len(sentence.split()))
              vector += (vec * tf_idf) 
              tf_idf_weight += tf_idf
      if tf_idf_weight != 0:
          vector /= tf_idf_weight
      tfidf_w2v_vectors.append(vector)
  return tfidf_w2v_vectors

class DataProcessor:
  # data -> List of strings
  def __init__(self, data):
    self.df = pd.DataFrame(data={"Description":data})
  
  def vectorize_data(self):
    ''' Return Tf-IDF weighted w2v values for list of sentences'''
    tfidf_vec = TfidfVectorizer()
    tfidf_vec.fit(self.df["Description"].values)
    tfidf_dict = dict(zip(tfidf_vec.get_feature_names(), list(tfidf_vec.idf_)))
    tfidf_words = set(tfidf_vec.get_feature_names())
    vectorize_data = tfdidf_w2v(self.df["Description"].values, tfidf_words, tfidf_dict)

    return vectorize_data


  def get_processed_data(self):
    ''' Preprocess data + Create new features + Vectorize text data'''
    self.df["Description"] = self.df["Description"].apply(pre_process_text)
    self.df["word_count"] = self.df["Description"].apply(lambda x: len(x.split(" ")))
    self.df["avg_word_length"] = self.df["Description"].apply(lambda x: np.mean([len(word) for word in x.split(" ")]))
    self.df["difficult_words_score"] = self.df["Description"].apply(lambda x: textstat.difficult_words(x))
    self.df["flesch_reading_ease"] = self.df["Description"].apply(lambda x: textstat.flesch_reading_ease(x))
    self.df["flesch_kincaid_grade"] = self.df["Description"].apply(lambda x: textstat.flesch_kincaid_grade(x))
    self.df["coleman_liau_index"] = self.df["Description"].apply(lambda x: textstat.coleman_liau_index(x))
    self.df["noun_count"] = self.df["Description"].apply(lambda x: count_POS(x,"noun"))
    self.df["pronoun_count"] = self.df["Description"].apply(lambda x: count_POS(x,"pronoun"))
    self.df["verb_count"] = self.df["Description"].apply(lambda x: count_POS(x,"verb"))
    self.df["adverb_count"] = self.df["Description"].apply(lambda x: count_POS(x,"adverb"))
    self.df["adjective_count"] = self.df["Description"].apply(lambda x: count_POS(x,"adjective"))

    vectorized_text_data = self.vectorize_data()
    self.df = self.df.drop(columns=["Description"])
    final_data = np.hstack((vectorized_text_data, self.df.values))

    return final_data

def final_function_1(data):
  '''
  Input : List of sentences,
  Output : Predictions for input sentences
  '''
  data = DataProcessor(data).get_processed_data()
  best_model = pickle.load(open('xgb_model.sav','rb'))
  predictions = best_model.predict(data)
  pred = [class_labels[x] for x in predictions]
  pred_labels = [class_label_names[x] for x in predictions]

  pred_prob = best_model.predict_proba(data)
  return pred_labels, pred_prob


print(os.listdir("."))