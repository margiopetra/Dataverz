# %%
import numpy as np
import pandas as pd
import re
import datetime


import zipfile
import pickle as pkl

import gensim
from gensim import utils
from gensim import downloader

from nltk import word_tokenize
from nltk import download
from nltk.corpus import stopwords
from string import punctuation

import matplotlib.pyplot as plt
#%%
from googletrans import Translator

from polyglot.detect import Detector
from polyglot.text import Text

#%%

from sklearn.linear_model import LogisticRegression
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

#%%
from sklearn.model_selection import train_test_split
from sklearn import linear_model

#%%
# open zipfile and read csv

# archive = zipfile.ZipFile("C://Users//Maria Petra//internship//ColombiaDocs//ColombiaDocs.zip",'r')
# csv = archive.open('Document (Node)_Document.csv')
# docs = pd.read_csv(csv, encoding='utf-8')

# make pkl file for faster processing

# with open('docs.pkl', 'wb') as handle:
#     pkl.dump(docs, handle)


# %%   
with open('docs.pkl','rb') as handle:
    documents = pkl.load(handle)


#%%

# Docment translation with googletrans API
#%%
def Punctuation_Removal(text):
    text = re.sub('[%s]' % re.escape(punctuation), '', text)
    return text

def doc_eng_traslation(text):

    # The function detects the language of a text and traslates it
    # in english if case it is not in english

    translator = Translator()  
    d = translator.detect(text)

    if d.lang=='en' in text:
        doc = text
    else:
        try:
            tr_text =  translator.translate(text, dest='en')
            tr_text = tr_text.text
            doc = (tr_text)
        except Exception as e:
            print(str(e))

    return doc


#%%


def Summary_laguage(documents):
    
    #Language detection function with polyglot

    new_series = []

    doc_Sum = Punctuation_Removal(str(documents.Summary))
    doc_Sum = ''.join(x for x in doc_Sum if x.isprintable())

    text_s = Text(str(doc_Sum)).language.code
    # print(text_s)
    new_series.append(str(text_s))

    # new_series = pd.DataFrame(new_series)
    return  np.array(new_series)

#%%
download('punkt')
download('stopwords')

stop_words = stopwords.words('english')


# %%
def Preprocessing(text):
    
    #Lower case, puctuation removal, tokenization, stop_word removal

    text = str(text)
    text = text.lower()
    text = Punctuation_Removal(text)
    doc = word_tokenize(text)
    doc = [word for word in doc if not word.isdigit()]
    doc = [word for word in doc if word not in stop_words]

    return doc

#%%
def norm_citation_count(df):

    # The function creates a normalized version of the Citation Count column
    # It subtracts the year of publication from the current year 
    # and divides it by the number of citations for each document

    now = datetime.datetime.now()
    current_year = now.year

    # for citation in df["Citation Count"]:
                    
    if df["Citation Count"] !=0 :
        norm = ((current_year + 1) - df["Year"])/df["Citation Count"]
    else:
        # print("ciation number" ,citation)
        norm = 0.0
    return norm

# %%
def Document_to_Vector (data):

    # The function tranforms the documents into word embedding by using gensims glove-wiki-gigaword-300 model

    vec=[]
    for doc in data:
        for word in doc:

            if word in model.vocab:
                vec.append(model[word])
            else:

                vec.append(np.zeros(model.vector_size))

    return np.sum(vec, axis=0)
    # return np.mean(vec, axis=0)


#%%
 # Some statistics about the Citation Count column
 # Maximum value, Mean number of citations, describe


print(documents['Citation Count'].describe())

#%%
citation_low = documents[(documents['Citation Count'] > 0) & (documents['Citation Count'] <= 2)]
citation_medium = documents[(documents['Citation Count'] > 2) & (documents['Citation Count'] <= 9)]
citation_hight = documents[documents['Citation Count'] > 9]

print("Citation low",len(citation_low))
print("Citation medium",len(citation_medium))
print("Citation high",len(citation_hight))



#%%
# data_partition["Norm Citation Count"] = norm_citation_count(data_partition)

#%%
documents["Norm Citation Count"] = documents.apply(norm_citation_count, axis = 1)
#%%
print(documents["Norm Citation Count"] .shape)
print(documents['Norm Citation Count'].describe())

#%%
from pandas.plotting import table

fig, ax = plt.subplots(1, 1)

table(ax, np.round(documents['Norm Citation Count'].describe(), 2))


documents['Norm Citation Count'].plot(ax=ax, legend=None)

#%%

documents['Norm Citation Count'].hist(bins = 20)
documents['Norm Citation Count'].plot.kde(ind=[0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5])
#%%
citation_low = documents[(documents['Norm Citation Count'] > 0) & (documents['Norm Citation Count'] <= 1)]
citation_medium = documents[(documents['Norm Citation Count'] > 1) & (documents['Norm Citation Count'] <= 4.39)]
citation_hight = documents[documents['Norm Citation Count'] > 4.39]

print("Citation low",len(citation_low))
print("Citation medium",len(citation_medium))
print("Citation high",len(citation_hight))

#%%

conditions = [
    (documents['Norm Citation Count'] <= 0),
    (documents['Norm Citation Count'] > 0) & (documents['Norm Citation Count'] <= 1),
    (documents['Norm Citation Count'] > 1) & (documents['Norm Citation Count'] <= 4.39),
    (documents['Norm Citation Count'] > 4.39),
    ]

values = ['zero','low', 'medium', 'high']

documents['Normalized Citation Class'] = np.select(conditions, values)

documents.head()

#%%
data = documents[(documents['Normalized Citation Class'] == "low") | (documents['Normalized Citation Class'] == "medium") |(documents['Normalized Citation Class'] == "high")]
print(data.shape)
print(documents.shape)

#%%

print(data["Norm Citation Count"] .shape)
print(data['Norm Citation Count'].describe())


#%%
data.loc[data["Summary"].isnull(),'Summary'] = data["Title"]

#%%
data["Summary Language"] = data.apply(Summary_laguage, axis = 1)
#%%
data = data[data['Summary Language'] == "en"]

#%%
print(data.shape)
print(documents.shape)

#%%
print(data["Summary"].iloc[1])
data["Summary Preprocessing"] = data["Summary"].apply(Preprocessing)
#%%
print(data["Summary Preprocessing"].iloc[1])



# Doc2Vec

# %%
model = gensim.downloader.load('glove-wiki-gigaword-300')


#%%
data["Summary Vectors"] = data["Summary"].apply(Document_to_Vector)

#%%


data['shapes'] = [x.shape for x in data["Summary Vectors"].values]
print(data.shape)
data = data[data['shapes']==(300,)]
print(data.shape)




# %%
doc_vectors_array = np.array(data["Summary Vectors"])
print(doc_vectors_array.shape)
doc_vectors_array = np.vstack(doc_vectors_array)
print(doc_vectors_array.shape)



#Classification
#%%
X_train, X_test, Y_train, Y_test = train_test_split(doc_vectors_array, data['Normalized Citation Class'], test_size=0.4, random_state=123)

#%%
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)

#%%
logreg = linear_model.LogisticRegression()
logreg.fit(X_train, Y_train)
predictions = logreg.predict(X_test)

#%%
print("Accuracy Score \n",accuracy_score(Y_test, predictions),"\n")
print("Confusion Matrix \n",confusion_matrix(Y_test, predictions),"\n")
print("Classification Report \n",classification_report(Y_test, predictions),"\n\n")


#Regression
#%%
X_train, X_test, Y_train, Y_test = train_test_split(doc_vectors_array, data['Norm Citation Count'], test_size=0.3, random_state=123)

regr = linear_model.LinearRegression()

#%%
regr.fit(X_train, Y_train)
y_pred = regr.predict(X_test)
# %%
# %%
print('Coefficients: \n', regr.coef_)
print('Mean squared error: %.2f' % mean_squared_error(Y_test, y_pred))

# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination: %.2f' % r2_score(Y_test, y_pred))

