#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import sys
import os
import json
import time
from datetime import date,datetime,timedelta
import csv
import re  # regular expressions (for playing with the text)
#import string
import nltk.data
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
from nltk.sentiment.vader import SentimentIntensityAnalyzer
#Sentiment intensity analyzer from nltk
sid = SentimentIntensityAnalyzer()
import gensim
from gensim import corpora, models
from gensim.models import ldamodel
from gensim.parsing.preprocessing import STOPWORDS # common english "stop words" -- a, the, etc.
from gensim.models.phrases import Phrases, Phraser
from collections import Counter

from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer #to tokenize glove dataset and documents
from keras.models import Sequential
from keras.layers import Input,Dense,Flatten,Embedding,Activation,Dropout,GlobalMaxPooling1D
from keras.layers.convolutional import Conv1D,MaxPooling1D
from keras.wrappers.scikit_learn import KerasClassifier #Keras wrapper for scikit learn (gridsearch)
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint, Callback


import sklearn
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelBinarizer
lb = LabelBinarizer()
from sklearn.metrics import precision_score, recall_score, f1_score
import sklearn.datasets as skds
from pathlib import Path
from hdfs import InsecureClient


#Get subjects of interests for each article.
def get_subjects(text):
    if len(text) == 0:
        return None
    else:
        lst1 = text.split(";")
        lst2 = []
        for sbj in lst1:
            try:
                #lst2.append(sbj.split("("))
                tmp_list = sbj.split("(")
                if int(tmp_list[1][:-2]) > 80:
                    lst2.append(tmp_list[0])
            except:
                pass
        if len(lst2) == 0:
            return None
        else:
            return ",".join(lst2)

swords=stopwords.words('english')  #stopwords from nltk
for i in STOPWORDS:   #STOPWORDS from gensim. Combining both to create a comprehensive list of stopwords.
    swords.append(i)
add = ['pm','jan','feb','mar','apr','may','jun','jul','aug','sep','dec','oct','nov','said','emailtoken','numbertoken','percenttoken', 'moneytoken', 'http', 'said', 'download_tabl', 'flow_ifram','countertoken', 'urltoken','uht']
swords.extend(add)


#Definitions of functions used in the program:
def remove_nonascii(text):
    return ''.join([i if ord(i) < 128 else '' for i in text])

def separate_hashtags(text):
    return set(part[1:] for part in text.split() if part.startswith("#"))
        
def replace_abbrevs(text):
    return re.sub(r'([a-zA-Z])([\'\-\.])(?=[a-zA-Z])', r'\1', text)

def replace_email(text):
    return re.sub(r'[\w\.\-]+@[\w\.\-]+', '', text)

def replace_urls(text):
    ##'''https://stackoverflow.com/questions/11331982/how-to-remove-any-url-within-a-string-in-python'''
    return re.sub('\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*',r'',text)

def replace_numbers(text):
    price_exp = re.compile(r"\$(\d*\,){,}\d+\.?\d*")
    pct_exp = re.compile(r'\b(\d*\,){,}\d+\.?\d*\%')
    counter_exp = re.compile(r"(\d*\,){,}\d+\.?\d*(st|nd|rd|th)s?")
    num_exp = re.compile(r"\b(\d*\,){,}\d+\.?\d*\b")
    text = re.sub(price_exp, '', text)
    text = re.sub(pct_exp, '', text)
    text = re.sub(counter_exp, '', text)
    return re.sub(num_exp, '', text)

def remove_punctuations(translator,text):  #Not used
    return ' '.join(text.translate(translator).split()).lower()

def read_corpus_basic(corp):  #Not used
    for doc in corp:
        yield [x for x in gensim.utils.simple_preprocess(doc, deacc=True)]

#Calls all the text processing functions above and cleans the text.
def clean(text): 
    Text = replace_email(text)
    Text = replace_urls(Text)
    Text = replace_abbrevs(Text)
    Text = replace_numbers(Text)
    Text = remove_nonascii(Text)
    return Text

#Removes stopwords, preprocesses to remove punctuations. Corp should be a list of documents
def read_corpus_with_stemming_and_SW_removal(corp):
    for doc in corp:
        yield [lemmatizer.lemmatize(x) for x in gensim.utils.simple_preprocess(doc, deacc=True)
                   if x.lower() not in swords]

#Takes the df column(series of documents) and returns a list of cleaned texts. 
#The tokens(cleaned words in each text) contain bigrams, trigrams and are separated by space.
def get_tokens(df):
    print('pre-processing with reg exp')
    # replace non-ascii characters, apostrophes, periods inside of words, urls, and numbers, using reg exp
    #apply 'clean' function on the text calls the other functions and cleans the data. 
    articles = df.Text.apply(clean)
    
    print('reading tokens from text')
    corp2 = list(read_corpus_with_stemming_and_SW_removal(articles))
    #print(corp2)
    
    print('adding bigrams')
    # identify bigrams in the text descriptions
    bigrams2 = gensim.models.phrases.Phrases(threshold=50)
    bigrams2.add_vocab(corp2)
    bigram_phraser2 = gensim.models.phrases.Phraser(bigrams2)
    
    print('adding trigrams')
    # we can apply the bigram phraser again to look for trigrams
    trigrams2 = gensim.models.phrases.Phrases(threshold=80)
    trigrams2.add_vocab(bigram_phraser2[corp2])
    trigram_phraser2 = gensim.models.phrases.Phraser(trigrams2)
    
    print('Done!')
    return [" ".join(trigram_phraser2[bigram_phraser2[tokens]]) for tokens in corp2]

#Builds a sequential model. Pass the required parameters when function is called and when the model is fit. Default values are below.
def define_model(embedding_layer,num_filters=64,kernel_size=3,optimizer='adam', dropout = 0.3, hidden_neurons=200,retrain=0): #,**kwargs
    model = Sequential()
    model.add(embedding_layer)
    model.add(Conv1D(filters=num_filters, kernel_size=kernel_size, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(hidden_neurons, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(1, activation='sigmoid'))   #dense 1 and sigmoid activation since its binary classification
    if retrain==1: #Once in a week, the model gets retrained with all the data
        #importing weights from saved model
        print('Importing weights from saved model')
        model.load_weights("best.weights.hdf5") #comment this for the first run
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['acc']) # compile the model
    return model

#integer encode and pad documents the training data. Similar to creating bag of words.
#docs should be a list of articles
#tokenizer is the tokenizer object created and fitted during training the data.
def encode_docs(tokenizer, max_length, docs):
    # integer encode the documents to replace the words with the corresponding interged from word index.
    encoded = tokenizer.texts_to_sequences(docs)
    #Now the docs are encoded with numbers but they are of different lengths. 
    #specify the max length and pad all the docs at the end (post) with zeroes if the lengh of docs is shorter.
    padded_docs = pad_sequences(encoded, maxlen=max_length, padding='post')
    return padded_docs

#Trains/retrains the model with X and Y data. Retrains if retrain = 1. Default is retrain = 0.
def retrain(X,Y,retrain=0):
    max_length = 400
    tokenizer = Tokenizer() 
    #X is List of news articles.Y are the corresponding labels.
    tokenizer.fit_on_texts(X)
    #Get the vocab size using the word index. 
    vocab_size = len(tokenizer.word_index) + 1 #(total number of unique words)
    
    #Read the glove file and create word embeddings
    embeddings_index = dict()
    f = open('Lexis_Output/glove.6B.100d.txt',encoding="utf8")  #100d -> 100 dimensions
    for line in f:
        values = line.split() #each word contains a word and its coefficients
        word = values[0]   #first element is the word
        coefs = np.asarray(values[1:], dtype='float32') #100 coeffs are after word. 
        embeddings_index[word] = coefs  #creates a dict of word and its coeffs. 
    f.close()
    print('Found %s word vectors.' % len(embeddings_index))
    
    #We now have the words and its coeffs. 
    #Using this index, we should create a matrix with weights for each word in training 
    MAX_SEQUENCE_LENGTH = max_length # 4000
    MAX_NB_WORDS = vocab_size # 29672
    EMBEDDING_DIM = 100  #100d
    word_index = tokenizer.word_index  #t is the tokenizer object.
    #print(len(word_index))
    
    #initiate zero weights can use np.random.random to initiate random weights
    embedding_matrix = np.zeros((vocab_size, EMBEDDING_DIM)) #matrix of 29672,100
    for word, i in word_index.items(): #iterate through the words in word_index
        embedding_vector = embeddings_index.get(word) #get the vector for the word from embeddings dict
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    #embeding layer is ready to be used as the input layer in our model.
    embedding_layer = Embedding(vocab_size,
                            EMBEDDING_DIM,weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,trainable=False)
    
    #encode input data using encode_docs function. Pass the tokenizer, max length and raw input data.
    Xtrain = encode_docs(tokenizer, max_length, X)
    #encode label data
    Ytrain = lb.fit_transform(Y)  #irrelevant is 0 and relevant is 1
    # define model
    model = define_model(embedding_layer,retrain=retrain)
    # fit network
    checkpointer = ModelCheckpoint(filepath='best.weights.hdf5', monitor='val_acc', verbose=1, save_best_only=True,mode='max')
    model.fit(Xtrain, Ytrain, validation_split=0.2, batch_size = 5, epochs=30, shuffle=True, callbacks=[checkpointer], verbose=0)    
    return tokenizer,model    

#For each news article in the file, encodes the document, classifies it as relevant/irrelevant and returns the label, squashed value
def classify(tokenizer,nn_model,text):
    max_length=400
    test_post = []
    test_post.append(text)
    Xtest = encode_docs(tokenizer,max_length, test_post)
    i = nn_model.predict(Xtest, verbose=0, steps=None)
    #i looks like [[0.026]]
    if i[0][0]>0.5:
        return (i[0][0],"Relevant")
    else:
        return (i[0][0],"Irrelevant")

#Input is the topics information obtained from topic modeling.
#Reformats the topics data into topic number, words and its prob as frequency and returns the info.
def get_topic_words(topics,num_topics):
    topic,words,freq,=[],[],[]
    for i in topics:
        for j in i[1]:
            words.append(j[0])
            freq.append(j[1]*1000)
    for i in range(num_topics):
        topic.extend(['Topic'+str(i+1)]*len(topics[0][1]))
    topic_info = pd.DataFrame()
    topic_info['Topic'] = pd.Series(topic)
    topic_info['Word'] = pd.Series(words)
    topic_info['Frequency'] = pd.Series(freq)
    return(topic_info)

#topic modeling on the cleaned tokens. Input is a list of cleaned news articles with lemmatized and nonstop words. 
def topic_model(articles,num_topics,num_words,passes):
    docs = []
    for text in articles:
        tokens=gensim.utils.simple_preprocess(text, deacc=True, min_len=3)
        #non_stop_tokens = [w for w in tokens if w not in swords]
        docs.append(tokens)
    dictionary = corpora.Dictionary(docs)
    corpus = [dictionary.doc2bow(text) for text in docs]
    gmodel = gensim.models.ldamodel.LdaModel(corpus=corpus,num_topics=num_topics,id2word=dictionary,passes=100)
    topics_data=gmodel.show_topics(num_topics=num_topics, num_words=num_words,formatted=False)
    topics_words=get_topic_words(topics=topics_data,num_topics=num_topics)
    return topics_words

def write_data(file_name,data):
    with open(file_name,"a") as f:
        wr = csv.writer(f,delimiter=",")
        wr.writerow(data)

#Checks if a file is available. Returns a boolean value.
def isFileAvailable(filename):
    try:
        #pd.read_csv('topics_words.csv')
        with open(filename, 'r') as test:
            return True
    except Exception as e:
        #print('Warning: ',e)
        print('Creating %s..' %filename)
        return False

#Can be used in future to read files from HDFS by submitting commands on command line.
# import subprocess 
# def run_cmd(args_list):
#         print('Running system command: {0}'.format(' '.join(args_list)))
#         proc = subprocess.Popen(args_list, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#         s_output, s_err = proc.communicate()
#         s_return =  proc.returncode
#         return s_return, s_output, s_err


#df=pd.read_csv('MasterCopy.csv',index_col=None, header=0)
#df['Tokens'] = get_tokens(df[['Text']])
#df['Classification']=df['Label'].apply(lambda x: 'Relevant' if x.lower().strip() != 'irrelevant' else 'Irrelevant')
#df[['Text','Tokens','Classification']].to_csv('MasterCopyBkp.csv',index_col=None)
#df[['Text','Tokens','Classification']].to_csv('MasterCopy.csv',index=False)


#Run this once to train the basic model using master data to create the tokenizer and model.
df=pd.read_csv('MasterCopy.csv', index_col=None,header=0)
X = list(df['Tokens'])
Y = list(df['Classification'])
tokenizer,nn_model=retrain(X,Y,retrain=0)


#Sample texts to check the model
#text1= 'this text has no relevant information see how gets classified. It might be disappointing if it cant classify the text appropriately but lets try to see if it can do any better than the basic machine learning models'
#text2= 'FORT LAUDERDALE, Fla. — It was the ballots, not the machines.To have a chance at overcoming Gov. Rick Scotts 12,603-vote lead in their Senate race, incumbent Bill Nelson, D-Fla., desperately needed a manual statewide recount to show that tens of thousands of votes here in Democrat-heavy Broward County had been misread by scanners.That didnt happen. And Nelsons chances of holding his Senate seat went from very slim to virtually nonexistent. As Florida recount deadline looms, Nelsons chances of victory dwindle NOV. 17, 201802:01 I dont see a path, said Steve Schale, a veteran Democratic strategist based in Florida. "Honestly, the path was never likely, but that doesnt diminish the need for a recount process, if for no other reason but to answer lingering questions — such as the undervotes in Broward, and provide certainty to all involved'
#text3= 'a buckhead sky rise marketed as a pioneer in transit connectedness could be proving again that marta proximity pays off in the modern atlanta economy the story atlanta plaza in buckhead is being renamed salesforce tower atlanta in light of the global software company commitment to add new jobs in its regional headquarters over the next five years'
#print(classify(tokenizer,nn_model,' '.join([lemmatizer.lemmatize(w) for w in gensim.utils.simple_preprocess(clean(text1),deacc=True)])))
#print(classify(tokenizer,nn_model,' '.join([lemmatizer.lemmatize(w) for w in gensim.utils.simple_preprocess(clean(text2),deacc=True)])))
#print(classify(tokenizer,nn_model,' '.join([lemmatizer.lemmatize(w) for w in gensim.utils.simple_preprocess(clean(text3),deacc=True)])))


def get_subjects(text):
    if text == None or len(text) == 0:
        return None
    else:
        lst1 = text.split(";")
        lst2 = []
        for sbj in lst1:
            try:
                #lst2.append(sbj.split("("))
                tmp_list = sbj.split("(")
                if int(tmp_list[1][:-2]) > 80:
                    lst2.append(tmp_list[0])
            except:
                pass
        if len(lst2) == 0:
            return None
        else:
            return ",".join(lst2)

if __name__ == '__main__':
    #This loop runs everyday. Program reads file from HDFS, processes it, creates required files and sleeps until next day.
    while True:
        #blank list to store all the data dictionaries from json file
        all_articles = []
        #temp dataframe to store formatted data after processing JSON file. 
        lexis_temp = pd.DataFrame()
        #get current data and print.
        curr_date=datetime.today()
        date_var=str(curr_date.year)+'-'+str(curr_date.month).zfill(2)+'-'+str(curr_date.day).zfill(2)
        print(curr_date)
        
        #Check for the day of the week and if retraining is required. Retrains every sunday.
        if curr_date.weekday() == 6: #checking for Sunday
            retrain_flag=1
        else:
            retrain_flag=0
        #Get the filenames for HDFS folder.
        client = InsecureClient('http://backend-0-3:50070', user='atl_sprint2018')
        files = client.list('/data/atl_sprint_2018/lexis_archive/')
        #Take the last vailable filename
        fjson = files[-1]   
        lastfile_date = datetime.strptime(fjson[6:14],'%Y%m%d')  #date of creation of last file. 
        delta = curr_date - lastfile_date  #calculate the difference between current date and the last file creation date.
        missing_data = 0  #counter to see how many records in the json file are with empty data.
        
#another method to get data from hdfs. Can be used for GPU processing
#       a=datetime.now()
#       date=str(a.year)+str(a.month).zfill(2)+str(a.day).zfill(2)
#       (ret, out, err)= run_cmd(['hadoop', 'fs', '-get', '/data/atl_sprint_2018/lexis_archive/lexis_%sT0000.json'
#                           %(date), './SocialMediaSprint/'])
#       if ret == 0:  #if return code is 0, file exists.
#           with open('lexis_%sT0000.json' %(date), 'r') as file1:
#       #Today’s file not found - If number of days between current date and last day’s file is > 0, 
#       #then today’s file is not found. Runs after 2 hours.
        
        if delta.days == 0:  #delta days = 0 ==> the last file was created today. Yay, we have data!
            #read the file from hdfs
            with client.read('/data/atl_sprint_2018/lexis_archive/' + fjson,encoding = 'utf-8',delimiter = '\n') as file1:
                for line in file1:  #each line is a json object (dictionary)
                    try:
                        news_article = json.loads(line)   
                        if news_article['Text'] != 'None' and len(news_article['Text'].split(' '))>=100:  #consider the data which has more than 50 words
                            all_articles.append(news_article)  #append individual news articles to the data list. 
                        else:
                            missing_data+=1
                    except:
                        continue
            #If there is data, processes the data and writes data into sentiment and topic files. 
            if len(all_articles) > 0:
                lexis_temp['Text']=pd.Series(list(map(lambda news_article:" ".join(news_article['Text'].split()),all_articles)))
                lexis_temp['City'] = pd.Series(list(map(lambda news_article: news_article['City'], all_articles)))
                lexis_temp['Comment_Count'] = pd.Series(list(map(lambda news_article: news_article['Comment_Count'], all_articles)))
                lexis_temp['Country'] = pd.Series(list(map(lambda news_article: news_article['Country'], all_articles)))
                lexis_temp['Data Source'] = pd.Series(list(map(lambda news_article: news_article['Data Source'], all_articles)))
                lexis_temp['Description'] = pd.Series(list(map(lambda news_article: news_article['Description'], all_articles)))
                lexis_temp['Favorite_Count'] = pd.Series(list(map(lambda news_article: news_article['Favorite_Count'], all_articles)))
                lexis_temp['Headlines'] = pd.Series(list(map(lambda news_article: news_article['Headlines'], all_articles)))
                lexis_temp['ID'] = pd.Series(list(map(lambda news_article: news_article['ID'], all_articles)))
                lexis_temp['Language'] = pd.Series(list(map(lambda news_article: news_article['Language'], all_articles)))
                lexis_temp['Location'] = pd.Series(list(map(lambda news_article: news_article['Location'], all_articles)))
                lexis_temp['Original Source'] = pd.Series(list(map(lambda news_article: news_article['Original Source'], all_articles)))
                lexis_temp['Screen_Name'] = pd.Series(list(map(lambda news_article: news_article['Screen_Name'], all_articles)))
                lexis_temp['Share_Count'] = pd.Series(list(map(lambda news_article: news_article['Share_Count'], all_articles)))
                lexis_temp['State'] = pd.Series(list(map(lambda news_article: news_article['State'], all_articles)))
                lexis_temp['Time'] = pd.Series(list(map(lambda news_article: news_article['Time'], all_articles)))
                lexis_temp['Time_Zone'] = pd.Series(list(map(lambda news_article: news_article['Time_Zone'], all_articles)))
                lexis_temp['URL'] = pd.Series(list(map(lambda news_article: news_article['URL'], all_articles)))
                lexis_temp['User_Name'] = pd.Series(list(map(lambda news_article: news_article['User_Name'], all_articles)))
                lexis_temp['User_id'] = pd.Series(list(map(lambda news_article: news_article['User_id'], all_articles)))
                lexis_temp['Tokens'] = pd.Series(get_tokens(lexis_temp[['Text']])) #get the cleaned tokens
                lexis_temp['Subject']=pd.Series(lexis_temp['Description'].apply(lambda x: x.get('Subject')))
                lexis_temp['Source'] = pd.Series(list(map(lambda news_article: news_article['Original Source'], all_articles)))
                lexis_temp['Drop_Criteria']=pd.Series(lexis_temp['Text'].apply(lambda x: x.replace(" ","") if not None else x))
                lexis_temp=lexis_temp.drop_duplicates(subset=['Drop_Criteria'],keep='first',inplace=False).reset_index(drop=True)
                
                
                
                #Classifying the new articles.
                print("classifying..")
                classification,confidence=[],[] 
                for t in lexis_temp['Tokens']:  #predict the output labels and the confidence of classification
                    #note1: the classification confidence we receive is a value squashed by sigmoid which is 
                    #likely to be close to 1 or 0. How do we get the actual probability?
                    conf,result=classify(tokenizer,nn_model,t) #tokenizer and nn_model get created and passed back after training/retraining.
                    classification.append(result)
                    confidence.append(conf)
                lexis_temp['Classification'] = pd.Series(classification)  #Add data to the dataframe
                lexis_temp['Confidence'] = pd.Series(confidence)                
                
                #Get only the rows with high confidence of classification to use them for for re-training.
                #Note2: Related to note1. How do we get the articles classified with highest confidence? 
                df_newdata=lexis_temp[(lexis_temp['Confidence'] > 0.99) | (lexis_temp['Confidence'] <= 0.000001)].reset_index(drop=True)
                #We accumulate the training samples for a week and retrain the model with old and new data together. 
                #we create a new file if NewTrainingData file does not exist. if it exists, we append the data to old file.
                file_exists=False
                file_exists = isFileAvailable('Lexis_Output/NewTrainingData.csv')
                if file_exists:
                    df_newdata[['Text','Tokens','Classification']].to_csv('Lexis_Output/NewTrainingData.csv',mode='a',index=False, header=False)
                else:
                    df_newdata[['Text','Tokens','Classification']].to_csv('Lexis_Output/NewTrainingData.csv',mode='a',index=False, header=True)
                
                               
                #Take the relevant news_articles and calculate sentiment and topics                
                print('Filtering the relevant news articles to calculate sentiment..')
                df_relevant=lexis_temp[lexis_temp['Classification']=='Relevant'].reset_index(drop=True)  #Select only the relevant news articles to evaluate topics and sentiments
                filedate = datetime.strftime((datetime.strptime(fjson[6:14],'%Y%m%d')-timedelta(days=1)),'%Y-%m-%d')
                df_relevant['Date'] = pd.Series([filedate]*len(df_relevant))
                
                ########################Getting the daily volume in a .CSV file###############################
                file_exists=False
                file_exists=isFileAvailable('Lexis_Output/Daily_News_Volume.csv')
                if file_exists:
                    df_relevant.groupby("Date").Text.count().reset_index(name="Volume").to_csv("Lexis_Output/Daily_News_Volume.csv",
                                                                        sep=",",header=False,mode='a',index=False)
                else:
                    df_relevant.groupby("Date").Text.count().reset_index(name="Volume").to_csv("Lexis_Output/Daily_News_Volume.csv",
                                                                        sep=",",header=True,mode='a',index=False) 
                
                ########################Getting the daily volume of Top 10 Sources in a .CSV file###############################
                df_relevant.groupby("Source").Text.count().reset_index(name="Volume").sort_index(by="Volume",
                    ascending=False)[:10].to_csv("Lexis_Output/Lexis_Source_Volume.csv",sep=",",header=True,mode='w',index=False)
                
                top_10_source= df_relevant.groupby("Source").Text.count().reset_index(name="Volume").sort_index(by="Volume",
                    ascending=False)[:10]
                
                ##################################Outputting the Data for Top 10 Sources###################################
                
                Output_Data=df_relevant[df_relevant['Source'].isin(list(top_10_source.Source))]
                Output_Data['Todays_Date']=date_var
                Output_Data[["Source","Text","Date","Todays_Date"]].to_csv('Lexis_Output/Top10_Source_Data.csv',sep=",",header=True,mode='w',index=False)
                
                ###########################################Subject Topics##################################################
                
                df_relevant['Subject_Topics'] = pd.Series(list(map(lambda news_article: get_subjects(news_article), df_relevant['Subject'])))
                
                topic_list=df_relevant['Subject_Topics'].tolist()
                topic_list=[x.split(',') for x in topic_list if x is not None]
                topic_list=[item.strip() for sublist in topic_list for item in sublist]
                topic_count=Counter(topic_list)
                topic_count=dict(topic_count)
                topic_count=pd.Series(topic_count, name='Frequency')
                topic_count.index.name='Topics'
                topic_count.reset_index(name="Frequency").to_csv('Lexis_Output/Topic_Subjects.csv',sep=",",header=True,mode='w',index=False)
                
                #prepare input for topic modeling.
                articles=list(df_relevant['Tokens'])
                num_topics = 5
                num_words = 10
                passes=100
                topics_words=topic_model(articles,num_topics,num_words,passes)
                topics_words['Date'] = pd.Series([filedate]*len(topics_words))
                
                
                
                #Write the reformatted topics data into a file. If the file is available, append the data, else write into a new file
                file_exists=False
                file_exists = isFileAvailable('Lexis_Output/topics_words.csv')
                if file_exists:
                    topics_words.to_csv('Lexis_Output/topics_words.csv', mode='a', header=False, index=False)
                else:
                    topics_words.to_csv('Lexis_Output/topics_words.csv', mode='a', header=True, index=False)
                
                
                
                #Calculate sentiment
                daily_filename = 'Lexis_Output/Daily_File_%s.csv' % (filedate)
                df_relevant['Sentiment'] = pd.Series([round(sid.polarity_scores(text)['compound'],1) for text in articles])
                #df_relevant['Date'] = pd.Series([filedate]*len(df_relevant))
                df_relevant[['Date', 'Text','Subject' ,'Sentiment', 'Classification']].to_csv(daily_filename,index=False)
                df_relevant['Sentiment_Category']= df_relevant['Sentiment'].apply(lambda x: 'Negative' if x<0 else 'Postive' if x>0 else 'Neutral')
                
                
                
                #Getting Overall Sentiment Category
                file_exists=False
                file_exists = isFileAvailable('Lexis_Output/Lexis_Overall_Sentiment.csv')
                if file_exists:
                    df_relevant.groupby(["Date","Sentiment_Category"]).Text.count().reset_index(name="Volume").to_csv("Lexis_Output/Lexis_Overall_Sentiment.csv",sep=",",header=False,mode='a',index=False)
                else:
                    df_relevant.groupby(["Date","Sentiment_Category"]).Text.count().reset_index(name="Volume").to_csv("Lexis_Output/Lexis_Overall_Sentiment.csv",sep=",",header=True,mode='a',index=False)
                
                
                
                #Getting Sentiment Category of Top 10 Sources
                sent_source=df_relevant[df_relevant['Source'].isin(list(top_10_source.Source))]
                sent_source.groupby(["Date","Source","Sentiment_Category"]).Text.count().reset_index(name="Volume").sort_index(by=["Source","Volume"],
                            ascending=False).to_csv("Lexis_Output/Lexis_Source_Sentiment.csv",sep=",",header=True,mode='w',index=False)
                
                
                if retrain_flag==1:
                    #if weekday ==6 (sunday), retrain the model for next day's use.
                    df_new=pd.read_csv('Lexis_Output/NewTrainingData.csv', index_col=None,header=0)  #training data accumulated over days
                    df_master=pd.read_csv('Lexis_Output/MasterCopy.csv', index_col=None, header=0) #master copy with manually labelled data
                    #append both and get new dataframe.
                    df_c=df_new[['Tokens', 'Classification']].append(df_master[['Tokens', 'Classification']],ignore_index=True)
                    X = list(df_c['Tokens'])
                    Y = list(df_c['Classification'])
                    print("Retraining with the new data..")
                    #return the new tokenizer and the retrained neural network model which will be used for prediction
                    tokenizer,nn_model=retrain(X,Y,retrain=retrain_flag)
                
                #put the program to sleep to run at 8 AM the next day.
                curr_date = datetime.now()
                tomorrow = (curr_date + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
                delta=tomorrow-datetime.now()  #remaining time for today
                print("Data processed. The program runs again at 8AM tomorrow.")
                time.sleep(delta.seconds +(3600*8))   #sleeps for remaining time today + 8 hours of next day.
            #If there is no data in today's input file, code runs next day at 8AM.
            else:
                print("No data in file. The program runs again at 8AM tomorrow")
                curr_date = datetime.now()
                tomorrow = (curr_date + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
                delta=tomorrow-datetime.now()
                #break
                #time.sleep(40)
                time.sleep(delta.seconds +(3600*8))
        #If today's file is not found, code runs after 2 hours. 
        else:
            print("File not found, will try again after 2 hours")
            curr_date = datetime.now()
            tomorrow = (curr_date + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
            delta=tomorrow-datetime.now()
            #break
            #If the file is not found ever after 10 PM, the program gives up and sleeps till 8 AM next day
            #In all the other cases, the program checks for file every 2 hours.
            if delta.seconds/3600 < 2:
                time.sleep(delta.seconds +(3600*8))
            else: 
                #time.sleep(10)
                time.sleep(7200)
