#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing required libraries
from wordcloud import WordCloud
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn import metrics
from sklearn.metrics import classification_report
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import re
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore")
sns.set(color_codes=True)
pd.set_option('display.max_rows', None, 'display.max_columns', None)


# In[2]:


df = pd.read_csv("news.csv")


# In[3]:


print("DATASET INFO")
print("-"*120)
print("\nShape of the dataset: ", df.shape)
print("\nColumns in the dataset: ", df.columns)
print("\nDisplaying dataset\n", df.head())
print("\nTotal NULL values found in the dataset: ", df.isnull().sum())
print("-"*120)


# In[4]:


# Creating a copy of the dataset
news = df.copy()


# In[5]:


# Checking length of each text article by character
length = []
[length.append(len(text)) for text in news['text']]
news['length'] = length
print('Minimum Length: ', min(news['length']), '\nMaximum Length: ', max(news['length']), '\nAverage Length: ',
      round(sum(news['length']) / len(news['length'])))


# In[6]:


# Minimum length is 1. Checking for outliers & removing them.
# Keeping 50 has minimum number of characters
print('Number of articles with less than 50 characters: ', len(news[news['length'] < 50]))


# In[7]:


# Displaying the data which has less than 50 characters in the text article
print(news['text'][news['length'] < 50])


# In[8]:


# Removing outliers to reduce overfitting & resetting index
news.drop(news['text'][news['length'] < 50].index, axis=0, inplace=True)
print('Minimum Length: ', min(news['length']), '\nMaximum Length: ', max(news['length']), '\nAverage Length: ',
      round(sum(news['length']) / len(news['length'])))


# In[9]:


# Resetting index to update row number in serial order
news.reset_index(inplace=True)
print(news.columns)


# In[10]:


# Dropping the new column 'Length'
news.drop(columns=['length', 'Unnamed: 0', 'index'], inplace=True)

# Checking for string containing only white_spaces
white_space_text = [idx for idx, text in enumerate(news.text.tolist()) if str(text).isspace()]
print("Numbers of rows where News Text contains white space(s): ", len(white_space_text))


# In[11]:


# Displaying the shape after removing outliers and empty data
print("Cleaned dataset shape: ", news.shape)
print(news.columns)


# ### DATA VISUALIZATION

# In[12]:


# Checking fot Target Class balance
sns.countplot(news.label)
plt.title("Class Data Balance Graph")
plt.xlabel("Target Class")
plt.ylabel("Target Count")
plt.savefig('class_balance.png')
plt.show()
print("As shown in the graph, Target classes are quite balanced")


# In[13]:


# Displaying Number of words in News Text
sns.displot(x=news['text'].str.split().apply(lambda a: len(a)), hue='label', data=news, rug=True)
plt.title("News Text Word Count")
plt.xlabel("Target Class")
plt.ylabel("Number of words in News Text")
plt.savefig('text_word_ct.png')
plt.show()


# In[14]:


# Displaying Common words in Fake News Text
all_keywords = " ".join(line for line in news[news.label == "FAKE"].text)
word_cloud = WordCloud(width=1250, height=625, max_font_size=350, random_state=42).generate(all_keywords)
plt.figure(figsize=(10, 5))
plt.title("Common words in Fake News Text", size=20)
plt.imshow(word_cloud)
plt.axis("off")
plt.savefig('wordcloud_fake.png')
plt.show()


# In[15]:


# Displaying Common words in Real News Text
all_keywords = " ".join(line for line in news[news.label == "REAL"].text)
word_cloud = WordCloud(width=1250, height=625, max_font_size=350, random_state=42).generate(all_keywords)
plt.figure(figsize=(10, 5))
plt.title("Common words in Real News Text", size=20)
plt.imshow(word_cloud)
plt.axis("off")
plt.savefig('wordcloud_real.png')
plt.show()


# ### Tokenization of the text data

# In[16]:


# Cleaning the data by filtering with the help of Stopwords & Stemming & converting into lowercase
stop = set(stopwords.words('english'))
ps = PorterStemmer()
corpus = []

for i in range(0, len(news)):
    review = re.sub('[^a-zA-Z]', ' ', news['text'][i])
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if word not in stop]
    review = ' '.join(review)
    corpus.append(review)


# In[17]:


X = corpus
y = news["label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


# In[18]:


# Initializing Vectorizer
tfidf = TfidfVectorizer(stop_words='english', max_df=0.7, max_features=9000, ngram_range=(1, 3))

# Fit and transform training set and transform test set
tfidf_train = tfidf.fit_transform(X_train).toarray()
tfidf_test = tfidf.transform(X_test)

# Setting up Passive Aggressive Classifier
pac = PassiveAggressiveClassifier(max_iter=1000)

# Fitting on the training set
pac.fit(tfidf_train, y_train)

# Predicting on the test set
pred = pac.predict(tfidf_test)


# In[19]:


# Displaying model metrics
print("Accuracy:   %0.2f" % metrics.accuracy_score(y_test, pred))


# In[20]:


# Displaying Classification Report & Confusion Metrics
print(f"Classification Report : \n\n{classification_report(y_test, pred)}")


# In[21]:


cm = metrics.confusion_matrix(y_test, pred)
classes = ['FAKE', 'REAL']
sns.heatmap(cm, annot=True, cmap='BuPu', fmt='g')
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks + 0.5, classes)
plt.yticks(tick_marks + 0.5, classes)
plt.xlabel("Actual Value")
plt.ylabel("Predicted Value")
plt.tight_layout()
plt.show()


# In[22]:


pickle.dump(pac, open('model_p1.pkl', 'wb'))
pickle.dump(tfidf, open('tfidf_vec.pkl', 'wb'))


# In[23]:


# Loading model and TFID vectorizer
pac_model = pickle.load(open('model_p1.pkl', 'rb'))
tf_vect = pickle.load(open('tfidf_vec.pkl', 'rb'))


# In[24]:


# Prediction for 5 random news from the given dataset
for i in range(1, 6):
    idx = np.random.choice(range(news.shape[0]))
    review = re.sub('[^a-zA-Z]', ' ', news['text'][idx])
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if word not in stop]
    review = ' '.join(review)

    val_pkl = tf_vect.transform([review]).toarray()
    print("-"*50)
    print("Actual Class value in row", idx, ": ", news['label'][idx])
    print("Pickled model prediction: ", pac_model.predict(val_pkl)[0])
    print("-"*50)


# In[ ]:




