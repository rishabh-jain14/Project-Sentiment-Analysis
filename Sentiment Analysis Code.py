#!/usr/bin/env python
# coding: utf-8

# # Sentiment Analysis
# 
# ## First Thing we need to do is import all the required libraries
# 
#     1. Data manipulation libraries like numpy, pandas for extraction of data from datasets and to perform data cleaning
#     2. Regex is used to recognize various patterns in the string/tweets
#     3. NLTK is used to perform feature extraction and text analysis (also, for twitter_samples database)
#     4. matplotlib.pyplot and seaborn for proper visualization of the data
#     5. stopwords to provide irrelevance to words like {"I", "you", "the", "him", "for", ...}
#     6. wordcloud for creating wordclouds to assess the frequently occuring words of dominant emotional significance.
# 

# In[ ]:


import pandas as pd
import numpy as np
import re

import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()
get_ipython().run_line_magic('matplotlib', 'inline')

import plotly.express as px

import nltk
from nltk.corpus import stopwords

import wordcloud
from wordcloud import WordCloud, STOPWORDS


# ## DataSet Collection and Pre-Processing
# 
# ### We will be using the Reviews.csv file from Kaggle’s Amazon Fine Food Reviews dataset to perform the analysis. This Dataset includes 568454 reviews with their individual scores (1 to 5)
# 
# #### (Mostly contains Positive Reviews)

# In[ ]:


food = pd.read_csv("Amazon Fine Foods/Reviews.csv")


# In[ ]:


food.head()


# In[ ]:


food.describe()


# Removing all the unnecessary columns leaving only the text itself and the score to it

# In[ ]:


food = food.drop(['Id', 'ProductId', 'Summary', 'UserId', 'ProfileName', 'Time', 'HelpfulnessNumerator', 'HelpfulnessDenominator'], axis = 1)


# Text — This variable contains the complete product review information
# 
# Score — The product rating provided by the customer

# In[ ]:


food.head()


# We must define a system that evaluates the text as positive/negative on the basis of the score given to it. We assume that the score 1 and 2 indicate a negative response (sad/angry/unsatisfied), 3 indicates a neutral response and 4 and 5 indicate a postove response (happy/calm/satisfied). So, we define a new column names 'Response'

# In[ ]:


conditions = [(food['Score'] > 3), (food['Score'] == 3), (food['Score'] < 3)]
food['Response'] = np.select(conditions, [1, 0, -1])
#1 for positive response
#0 for neutral response
#-1 for negative response

food.head()


# Now, we will take a look at the variable “Score” to see if majority of the customer ratings are positive or negative. Since on describing the data, we can see that the mean of the Score column is more than 4, it means that majority of the customer ratings are postiive. We can visualize this using the Plotly library

# In[ ]:


fig = px.histogram(food, x = "Score")
fig.update_layout(title_text='Product Score')
fig.show()


# Clearly, Postitive responses are way more than the negative ones. (Score=5 bar is sky-high)
# 
# Since, we have the required response target variable for each review now, we do not need the Score column; so, we remove that.
# 
# Finally, Food Reviews Dataset appears like this.

# In[ ]:


del food['Score']

food.head()


# ### We will also be using First GOP Debate Twitter Tweets Data from Kaggle as well since it has a higher number of responses showing negative emotion. This dataset contains over 10000+ tweets maintaining a good complex mix of english vocabulary and informal criticism.
# 
# #### (Mostly Contains Negative Responses)

# In[ ]:


tweets = pd.read_csv("Twitter Sentiments/Sentiment.csv")
tweets.head()


# In[ ]:


tweets.describe()


# Removing all the unnecessary columns leaving only the tweet itself and the designated sentiment with it.

# In[ ]:


tweets = tweets[["sentiment", "text"]]


# Just like dataframe "food", We must define each tweet as +1 (Positive), -1 (Negative) and 0 (Neutral). So, we define a new column named 'Response'

# In[ ]:


tweets_conditions = [(tweets['sentiment'] == "Positive"),
                     (tweets['sentiment'] == "Negative"),
                     (tweets['sentiment'] == "Neutral")]
tweets['Response'] = np.select(tweets_conditions, [1, -1, 0])

tweets.head()


# Now, we do not need the column "sentiment", similar to the Food Reviews Dataframe. Also, we need to equate the number of columns and their names, in every dataframe used, before concatenating them

# In[ ]:


del tweets["sentiment"]
tweets = tweets.rename(columns = {"text": "Text"})

tweets.head()


# ## Data Cleaning
# 
# We will be using the text data to come up with predictions. First, we need to remove all punctuation and digits from the data.

# In[ ]:


def remove_punctuation_and_digits(text):
    final = "".join(u for u in text if u not in ("?", ".", ";", ":", ',', "!",'"', "'", '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'))
    return final

food['Text'] = food['Text'].apply(remove_punctuation_and_digits)
tweets['Text'] = tweets['Text'].apply(remove_punctuation_and_digits)

# will take time


# Now, we have to work on removing all the hashtags, RTs, mentions and links from the dataset to transform it into just a string of words.
# 
# Furthermore, we also need to remove all the stopwords from the data so as to focus on all the relevant words only, to determine the sentiment of the string.

# In[ ]:


def remove_text_biome(text):
    stopwords_set = set(stopwords.words("english"))
    words_filtered = [e.lower() for e in text.split() if len(e) >= 3]
    words_cleaned = [word for word in words_filtered 
                     if 'http' not in word and not word.startswith('@') and not word.startswith('#') and word != 'RT']
    words_without_stopwords = [word for word in words_cleaned if not word in stopwords_set]
    return " ".join(words_without_stopwords)

food['Text'] = food['Text'].apply(remove_text_biome)
tweets['Text'] = tweets['Text'].apply(remove_text_biome)


# In[ ]:


final_data = pd.concat([food, tweets])


# In[ ]:


final_data


# ## Data Visualization
# 
# #### Let us visualize the dataset for Food Reviews for better insight of what we need to analyze for text analysis
# 
# We use stopwords which is a set including all the irrelevant grammatically-involved words in a sentence, which doesnt provide any impact to the sentiment of the text. We use stopwords to remove all the irrelevant words from the text first. Domain specific irrelevant words like "br", "href" can be added as well explicitly to the stopwords set.
# 
# Now, we can create some wordclouds to see the most frequently used "relevant" words in the reviews.

# #### WordCloud of all reviews Combined

# In[ ]:


stop_words = set(stopwords.words('english'))
stop_words.update(["br", "href"])

t_matter = " ".join(review for review in food.Text)
wordcloud = WordCloud(stopwords = stop_words).generate(t_matter)

plt.imshow(wordcloud, interpolation = 'bilinear')
plt.axis("off")
plt.savefig("wordcloud_all.png")
plt.show()

# may take some time to work since it will process each review for different words.


# #### Wordclouds can be used to identify which words can be proved helpful in identifying the sentiment of that review.
# 
# Now, we need to build two wordclouds, one of them will contain the words used in postive sentiments and the other will contain the words in negative sentiments.
# 
# To do this, we can create different dataframes for positive, neutral and negative responses respectively.

# In[ ]:


food_positive = food[food["Response"] == 1]
food_negative = food[food["Response"] == -1]
food_neutral = food[food["Response"] == 0]


# We will also use the colum "Text" for better acquisation of the review's sentiment and, efficient and more practical analysis
# 
# ### WordCloud for Postitive Responses

# In[ ]:


positive = " ".join(str(review) for review in food_positive.Text)
wordcloud_positive = WordCloud(stopwords = stop_words).generate(positive)

plt.imshow(wordcloud_positive, interpolation = 'bilinear')
plt.axis("off")
plt.savefig("wordcloud_pos.png")
plt.show()


# ### WordCloud for Negative Responses

# In[ ]:


negative = " ".join(str(review) for review in food_negative.Text)
wordcloud_negative = WordCloud(stopwords = stop_words).generate(negative)

plt.imshow(wordcloud_negative, interpolation='bilinear')
plt.axis("off")
plt.savefig("wordcloud_neg.png")
plt.show()


# ### WordCloud for Neutral Responses

# In[ ]:


neutral = " ".join(str(review) for review in food_neutral.Text)
wordcloud_neutral = WordCloud(stopwords = stop_words).generate(neutral)

plt.imshow(wordcloud_neutral, interpolation="bilinear")
plt.axis("off")
plt.savefig("wordcloud_neut.png")
plt.show()


# On analysing the wordclouds, we can see that the words "good" and "great" are redundent in every type of response. Therefore, for better accuracy of the model, we can negate them from all.

# In[ ]:


stop_words = set(stopwords.words('english'))
stop_words.update(['br', 'href', 'good', 'great'])

#New WordClouds without 'good' and 'great'

pos = " ".join(str(review) for review in food_positive.Text)
wordcloud_pos = WordCloud(stopwords = stop_words).generate(pos)
neg = " ".join(str(review) for review in food_negative.Text)
wordcloud_neg = WordCloud(stopwords = stop_words).generate(neg)
neut = " ".join(str(review) for review in food_neutral.Text)
wordcloud_neut = WordCloud(stopwords = stop_words).generate(neut)


# In[ ]:


#Visualizing all wordclouds after removal of the words 'great' and 'good'

plt.imshow(wordcloud_pos, interpolation = 'bilinear')
plt.axis("off")
plt.savefig("wordcloud_positive.png")
plt.title("Positive Words")
plt.show()
plt.imshow(wordcloud_neg, interpolation='bilinear')
plt.axis("off")
plt.savefig("wordcloud_negative.png")
plt.title("Negative Words")
plt.show()
plt.imshow(wordcloud_neut, interpolation="bilinear")
plt.axis("off")
plt.savefig("wordcloud_neutral.png")
plt.title("Neutral Words")
plt.show()


# ##### As seen above in the above worclouds
# 
# The positive sentiments' wordcloud was full of positive words, such as “love”, “best”, and “delicious.”
# 
# The negative sentiments' wordcloud was filled with mostly negative words, such as “disappointed”, "buy" and “yuck.”
# 
# The words “good” and “great” initially appeared in the negative sentiment word cloud, despite being positive words. This is probably because they were used in a negative context, such as “not good.” Due to this, I have removed those two words from the wordclouds.

# ## Building the Model

# Finally, we can build the sentiment analysis model.
# This is a classification task, so we will train a simple logistic regression model to do it.
# This model will take textual data in as input. It will then come up with a prediction on whether the review is positive, neutral or negative.

# ### Dataset Assignment
# 
# We will now split the data frame into train and test sets. 80% of the data will be used for training, and 20% will be used for testing.

# In[ ]:


# random split train and test data

indices = final_data.index
final_data['r_number'] = np.random.randn(len(indices))
train_data = final_data[final_data['r_number'] <= 0.8]
test_data = final_data[final_data['r_number'] > 0.8]


# In[ ]:


train_data


# In[ ]:


test_data


# ### Create a Bag-Of-Words
# 
# Now, we shall use a Tfid Vectorizer from the Scikit-learn library.
# 
# The Tfid Vectorizer will transform the text in our data frame into a bag of words model, which will contain a sparse matrix of integers. The number of occurrences of each word will be counted and stored.
# 
# We will need to convert the text into a bag-of-words model since the logistic regression algorithm cannot understand human text.

# In[ ]:


# TfidVectorizer Implementation

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(token_pattern=r'\b\w+\b')

train_matrix = vectorizer.fit_transform(train_data['Text'])
test_matrix = vectorizer.transform(test_data['Text'])


# ### Importing the Logistic regression model from scikit-learn

# In[ ]:


# Logistic Regression

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(max_iter = 1000)

# max_iter = 1000 or mode for better convergence of the model's accuracy, will extend if made higher,
# but it already has reached the convergence range.


# ### Assign the features and Targets to their respective variables

# In[ ]:


X_train = train_matrix
X_test = test_matrix

y_train = train_data['Response']
y_test = test_data['Response']


# ### Fit the data to the LR model and Predict the outcome of the model

# In[ ]:


lr.fit(X_train,y_train)

predictions = lr.predict(X_test)


# Now, we have trained the logistics regression model for sentiment analysis.

# ## Testing the Model
# 
# Let's Test the model now using confusion matrix and classification reports

# In[ ]:


# find accuracy, precision, recall:

from sklearn.metrics import confusion_matrix,classification_report

new = np.asarray(y_test)

confusion_matrix(predictions,y_test)


# In[ ]:


print(classification_report(predictions,y_test))


# Accuracy improvement Statistics:
# 
# Using Reviews Dataset Only using Naive Bayes classifier: 76%
# 
# Using Reviews Dataset Only using Logistic regression: 86%
# 
# Using Reviews Dataset and Twitter Dataset using Logistic regression: 88%
# 
# 
# 
# As we can assess from the above classification report and statistics, the Accuracy for the above sentiment analysis model is around 88%

# ## Model Showcase

# #### Let's Consider an example String (Review)

# In[ ]:


l = ["I'm so happy tonight!", "You are an awful person, Matt!", "Jeez, Control yourself man!"]
example_x = pd.Series(l)
example_x


# #### We follow the folllowing operations then:
# 
# 1. Extract relevant feature(s) from the normal text using the pre-processing functions
# 2. Use TfidVectorizer to define a relatively organized list of those features with a sparse matrix containg about 30000+ columns and rows for vocabulary analysis
# 3. when the pre-processing is completed, we feed the input to the model to predict the sentiment.

# Vectorizer used: TfidVectorizer and Count Vectorizer

# In[ ]:


example_x = example_x.apply(remove_punctuation_and_digits)
example_x = example_x.apply(remove_text_biome)
example_x


# In[ ]:


example_fx = vectorizer.transform(example_x)
example_p = lr.predict(example_fx)


# Now, we print the resultant sentiment, as per the prediction and designation: 1(+ve),  -1(-ve),  0(±ve)

# In[ ]:


for x in range(len(example_p)):
    if example_p[x] == 1:
        print("The Sentiment for the text: \'" + l[x] + "\' appears to be Positive")
    elif example_p[x] == -1:
        print("The Sentiment for the text: \'" + l[x] + "\' appears to be Negative")
    else:
        print("The Sentiment for the text: \'" + l[x] + "\' appears to be Neutral")

