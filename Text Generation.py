#!/usr/bin/env python
# coding: utf-8

# In[1]:


import random
import pickle
import numpy as np
import pandas as pd
from nltk.tokenize import RegexpTokenizer
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Activation
from tensorflow.keras.optimizers import RMSprop


# In[2]:


text_df = pd.read_csv('fake_or_real_news.csv')
text = list(text_df.text.values)
joined_text = " ".join(text)


# In[3]:


partial_text = joined_text[:100000]


# In[4]:


tokenizer = RegexpTokenizer(r"\w+")
tokens = tokenizer.tokenize(partial_text.lower())


# In[5]:


unique_tokens = np.unique(tokens)
unique_token_index = {token: idx for idx, token in enumerate(unique_tokens)}


# In[6]:


n_words = 10
input_words = []
next_words = []

for i in range(len(tokens) - n_words):
    input_words.append(tokens[i:i + n_words])
    next_words.append(tokens[i + n_words])


# In[7]:


x = np.zeros((len(input_words), n_words, len(unique_tokens)), dtype=bool)
y = np.zeros((len(next_words), len(unique_tokens)), dtype=bool)


# In[8]:


for i, words in enumerate(input_words):
    for j, word in enumerate(words):
        x[i,j, unique_token_index[word]] = 1
    y[i, unique_token_index[next_words[i]]] = 1


# In[9]:


model = Sequential()
model.add(LSTM(128, input_shape=(n_words, len(unique_tokens)), return_sequences=True))
model.add(LSTM(128))
model.add(Dense(len(unique_tokens)))
model.add(Activation("softmax"))


# In[10]:


model.compile(loss="categorical_crossentropy", optimizer=RMSprop(learning_rate=0.01), metrics=["accuracy"])
model.fit(x,y, batch_size=128, epochs=30, shuffle=True)


# In[11]:


model.save("textgen.keras")


# In[12]:


model = load_model("textgen.keras")


# In[20]:


def predict_next_words(input_text,n_best):
    input_text = input_text.lower()
    x = np.zeros((1,n_words, len(unique_tokens)))
    for i, word in enumerate(input_text.split()):
        x[0,i, unique_token_index[word]] = 1
    prediction = model.predict(x)[0]
    return np.argpartition(prediction, -n_best)[-n_best:]


# In[38]:


possible = predict_next_words("The president of the US has announced that", 10) 


# In[39]:


print([unique_tokens[idx] for idx in possible])


# In[21]:


def generate_text(input_text, text_length, creativity=3):
    word_sequnce = input_text.split()
    current = 0
    for _ in range(text_length):
        sub_squence = " ".join(tokenizer.tokenize(" ".join(word_sequnce).lower())[current:current+n_words])
        try:
            choice = unique_tokens[random.choice(predict_next_words(sub_squence, creativity))]
        except:
            choice = random.choice(unique_tokens)
        word_sequnce.append(choice)
        current +=1
    return " ".join(word_sequnce)


# In[43]:


input_texts = input("Enter the incomplete sentence: ")
text_lengths = int(input("How many word your want: "))
creativity_level = int(input("How much creatiity in number format: "))
generate_text(input_texts, text_lengths, creativity_level)


# In[19]:


model = load_model("textgen.keras")


