import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string 
import math
import nltk
from nltk.corpus import stopwords
from nltk.util import ngrams
from collections import Counter
from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer
# from wordcloud import WordCloud
# from textwrap import wrap
# import spacy 

# nlp = spacy.load('en_core_web_sm')

nltk.download('stopwords')
stop=set(stopwords.words('english'))

data  = pd.read_csv(r'C:\Users\Tejas\Desktop\Capstone\Datasets\davidson2017.csv')

# data = data[['tweet','class', 'hate_speech','offensive_language','neither']]

data = data[['tweet','hate_speech','offensive_language','neither']]

print("============= BEFORE CLEANING =============")

print("\nShape\n")
print(data.shape, "\n")
print("Describe\n")
print(data.describe(),"\n")
print("Null Values\n")
print(data.isnull().sum(), "\n")
print("Duplicate\n")
print(data.duplicated().sum(), "\n")
print("Correlation Matrix\n")
print(data.corr(), "\n")


print("============= AFTER CLEANING =============")

data['tweet_lower']=data['tweet'].apply(lambda x: x.lower())
data['tweet_lower']=data['tweet_lower'].apply(lambda x: re.sub('[%s]' % re.escape(string.punctuation), '', x))
data['tweet_lower']=data['tweet_lower'].apply(lambda x: re.sub(' +',' ',x))


print("Shape", data.shape)
print("Describe", data.describe())

print("Null Values", data.isnull().sum())
print("Duplicate", data.duplicated().sum())
print("Correlation Matrix", data.corr())



# corpus=[]
# new= df['tweet_lower'].str.split()
# new=new.values.tolist()
# corpus=[word for i in new for word in i]
# dic=defaultdict(int)
# for word in corpus:
#     if word in stop:
#         dic[word]+=1

# top = sorted(dic.items(), key= lambda x:x[1], reverse=True)[:10]
# x,y= zip(*top)
# plt.bar(x,y)
# plt.show()

# print(dic)

# counter=Counter(corpus)
# most=counter.most_common()

# x, y= [], []
# for word,count in most[:70]:
#     if (word not in stop):
#         x.append(word)
#         y.append(count)
        
# sns.barplot(x=y,y=x)
# plt.show()

# def plot_top_ngrams_barchart(text, n=2):
#     stop=set(stopwords.words('english'))

#     new= text.str.split()
#     new=new.values.tolist()
#     corpus=[word for i in new for word in i]

#     def _get_top_ngram(corpus, n=None):
#         vec = CountVectorizer(ngram_range=(n, n)).fit(corpus)
#         bag_of_words = vec.transform(corpus)
#         sum_words = bag_of_words.sum(axis=0) 
#         words_freq = [(word, sum_words[0, idx]) 
#                       for word, idx in vec.vocabulary_.items()]
#         words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
#         return words_freq[:10]

#     top_n_bigrams=_get_top_ngram(text,n)[:10]
#     x,y=map(list,zip(*top_n_bigrams))
#     sns.barplot(x=y,y=x)
#     plt.show()

# plot_top_ngrams_barchart(df['tweet_lower'],3)
# df['lemmatized']=df['tweet_lower'].apply(lambda x: ' '.join([token.lemma_ for token in list(nlp(x)) if (token.is_stop==False)]))
