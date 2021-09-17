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

stemmer = nltk.SnowballStemmer("english")
stopword=set(stopwords.words('english'))

nltk.download('stopwords')
stop=set(stopwords.words('english'))

data  = pd.read_csv(r'C:\Users\Tejas\Desktop\Capstone\Datasets\davidson2017.csv')

df = data[['tweet','class']]

def clean_text(text):
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = [word for word in text.split(' ') if word not in stopword]
    text=" ".join(text)
    text = [stemmer.stem(word) for word in text.split(' ')]
    text=" ".join(text)
    return text

df['tweet_lower']=data['tweet'].apply(clean_text)

corpus=[]
new= df['tweet_lower'].str.split()
new=new.values.tolist()
corpus=[word for i in new for word in i]

def plot_top_ngrams_barchart(text, n=2):
    stop=set(stopwords.words('english'))

    def _get_top_ngram(corpus, n=None):
        vec = CountVectorizer(ngram_range=(n, n)).fit(corpus)
        bag_of_words = vec.transform(corpus)
        sum_words = bag_of_words.sum(axis=0) 
        words_freq = [(word, sum_words[0, idx]) 
                      for word, idx in vec.vocabulary_.items()]
        words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
        return words_freq

    top_n_bigrams=_get_top_ngram(text,n)[:30]
    x,y=map(list,zip(*top_n_bigrams))
    sns.barplot(x=y,y=x)
    plt.show()

for num in range(1,5):
    plot_top_ngrams_barchart(df['tweet_lower'],num)
