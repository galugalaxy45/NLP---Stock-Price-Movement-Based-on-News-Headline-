#!/usr/bin/env python
# coding: utf-8

# # <span style = "color:green"> Stock Price Movement Based on News Headline </span>

# ***

# There are various kinds of new articles and based on that the stock price fluctuates. We will analyze the news heading using sentiment analysis using NLP and then we will predict if the stock will increase or decrease. It is all about stock sentiment analysis

# ### Content

# This dataset is a combination of world news and stock price available on Kaggle. There are 25 columns of top news headlines for each day in the dataframe, Date, and Label (dependent feature). Data range from 2008 to 2016 and the data frame 2000 to 2008 was scrapped from yahoo finance. Labels are based on the Dow Jones Industrial Average stock index.

# * Class 1 - Stock Price Increased
# * Class 0 - Stock Price Decreased

# ## Let's Dive into it

# ### Import necessary Libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import re


# ### Read 'stocknews.csv' and store it in a dataframe

# In[2]:


data = pd.read_csv('stocknews.csv', encoding = 'ISO-8859-1')


# ### View head

# In[3]:


data.head()


# ### View unique values in Label

# In[4]:


data['Label'].unique()


# ### Check for null values

# In[5]:


data.isna().sum()


# ### Drop null rows

# In[6]:


data.dropna(inplace = True)


# ### Check for duplicates

# In[7]:


data.duplicated().sum()


# ### Check info of the dataset

# In[8]:


data.info()


# ### Plot a countplot of labels

# In[9]:


sns.countplot(y = 'Label', data = data)
plt.show()


# ### Combine all news headlines into a single text and store it in a list

# In[10]:


headlines = []
for row in range(0,len(data.index)):
    headlines.append(' '.join(str(x) for x in data.iloc[row,2:27]))


# ### View one of the combined text

# In[11]:


headlines[0]


# ### Store the list as a column in our original Dataframe. (Name it Headlines)

# In[12]:


data['Headlines'] = headlines


# In[13]:


data.head()


# ### Only keep Healines and Label in our dataset

# In[14]:


data = data[['Headlines','Label']]
data.head()


# ### Create a wordcloud of headlines

# In[18]:


from wordcloud import WordCloud


# In[19]:


plt.figure(figsize = (12,16))
wordcloud = WordCloud(background_color = 'white').generate(' '.join(data['Headlines'].tolist()))
plt.imshow(wordcloud)
plt.show()


# ### Create a function to preprocess the text
# * Remove any special characters (Symbols)
# * Remove any stopwords
# * Lemmatize the words
# * Convert all words to lowercase

# In[20]:


from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords


# In[21]:


def preprocess(sentence):
    #removes all the special characters and split the sentence at spaces
    text = re.sub(r'[^0-9a-zA-Z]+',' ',sentence).split()
    
    # converts words to lowercase and removes any stopwords
    words = [x.lower() for x in text if x not in stopwords.words('english')]
    
    # Lemmatize the words
    lemma = WordNetLemmatizer()
    word = [lemma.lemmatize(word,'v') for word in words ]
    
    # convert the list of words back into a sentence
    word = ' '.join(word)
    return word


# In[22]:


preprocess(headlines[0])


# ### Apply the function on Headlines column

# In[23]:


data['Headlines'] = data['Headlines'].apply(preprocess)


# In[24]:


data.head()


# ### Print some of the processed text

# In[25]:


for i in range(5):
    print(data['Headlines'][i])


# ### Assign input and Target Variables

# * X - Headlines
# * y - Label

# In[26]:


X = data['Headlines']
y = data['Label']


# ### Apply TfidfVectorizer

# In[27]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[28]:


vectorizer = TfidfVectorizer(ngram_range=(2,2))


# In[29]:


X = vectorizer.fit_transform(X)


# In[30]:


type(X)


# ### Split the dataset into training and Testing set

# In[31]:


from sklearn.model_selection import train_test_split


# In[32]:


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2)


# ### Check the shape of X_train and X_test

# In[33]:


X_train.shape


# In[34]:


X_test.shape


# ### Create an SVM model and Train it

# In[35]:


from sklearn.svm import SVC


# In[36]:


model = SVC()


# In[37]:


model.fit(X_train, y_train)


# ### Check the score of the training set

# In[38]:


model.score(X_train, y_train)


# ### Make predictions with X_test

# In[39]:


prediction = model.predict(X_test)


# ### Check the accuracy of our prediction

# In[40]:


from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# In[41]:


accuracy_score(y_test, prediction)


# ### Create confusion matrix and plot it on a heatmap

# In[42]:


sns.heatmap(confusion_matrix(y_test,prediction),annot = True, fmt = 'd')
plt.show()


# ### Print Classification Report

# In[43]:


print(classification_report(y_test,prediction))


# ***
