# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 13:42:59 2023

@author: PauloAndrade
"""

#BIBLIOTECAS UTILIZADAS
import plotly.express as px
import pandas as pd
import numpy as np
import seaborn as sns
import openpyxl
import matplotlib.pyplot as plt
import nltk
import string
import re
from wordcloud import WordCloud
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import *
from nltk import WordNetLemmatizer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
# nltk.download()

#CARREGAR BASE INICIAL
base_inicial = pd.read_csv('news_headlines.csv')


#PADRONIZAR DATAFRAME
base_treino = base_inicial.rename(columns={'tags':'label','post':'text'})

#EXIBIR QTD DE REGISTRO
print(f'Qtd de Registro: {len(base_treino)}')

#EXIBIR QTD DE CATEGORY
print(f"Qtd de Categorias: {base_treino['label'].nunique()}")

plt.figure(figsize= (15,7))
plt.subplot(1,2,1)
sns.countplot(x = 'label', data = base_treino)
plt.title('Distribuição de Registros')
plt.xlabel('Label')
plt.ylabel('Quantidade')
plt.subplot(1,2,2)
plt.pie(base_treino['label'].value_counts() , autopct = '%1.1f%%')
plt.legend(base_treino['label'].unique())
plt.title ('%Distribuição de Categorias')
plt.show()

#NUVEM DE PALAVRAS
wc = WordCloud(background_color =  'white')
wc.generate(str(base_treino['text']))
plt.imshow(wc, interpolation = 'bilinear')
plt.title('Nuvem Antes do Processamento')
plt.axis('off')
plt.show()


def pre_processamento(texto):
    # LOWER text
    texto = texto.lower()
    # PONTUAÇÃO --> remove todos as pontuações dos textos
    texto = re.sub(r'[^a-zA-Z]+', ' ', texto)
    # TOKEN --> dividir frases em palavras ou tokens individuais
    tokens = word_tokenize(texto)
    # REMOVE STOPWORDS - remove as palavras mais comuns e sem significado do texto
    tokens = [word for word in tokens if word not in stopwords.words('portuguese')]
    # LEMMATIZA
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    texto_processado = ' '.join(tokens)
    return texto_processado

 
#APLICANDO O PRE PROCESSAMENTO NA BASE INICIAL
base_treino['texto_processado'] = base_treino['text'].apply(pre_processamento)

#NUVEM DE PALAVRAS APÓS PRÉ PROCESSAMENTO
wc = WordCloud(background_color =  'white')
wc.generate(str(base_treino['text']))
plt.imshow(wc, interpolation = 'bilinear')
plt.title('Nuvem após Processamentos')
plt.axis('off')
plt.show()

#--------------------------------------VECTORIZE COM NAIVE BAYES-----------------------------------------------------#

#BAG OF WORDS - separa palavra por palavra e apresenta a sua ocorrencia no texto
vectorizer = CountVectorizer()
vec_transform = vectorizer.fit_transform(base_treino['text'])  #treinamento
data_count_vector = np.array(vec_transform.todense(), dtype='float32')

#
x_train , x_test , y_train , y_test = train_test_split(
    data_count_vector , base_treino['label'], test_size = 0.2, random_state = 0)


naive = MultinomialNB()
modelo = naive.fit(x_train,y_train)


predict_train = modelo.predict(x_train)
predict_test = modelo.predict(x_test)

print('\nClassification Reporte Train - Vectorizer com Naiv Bayes \n',metrics.classification_report(y_train, predict_train))
print('\n')
print('Confusion Matrix \n',metrics.confusion_matrix(y_train, predict_train))
print('\n')
print('Accuracy of train : {0:0.3f}'.format(metrics.accuracy_score(y_train,predict_train)))

print('\nClassification Reporte Test - Vectorizer com Naiv Bayes \n',metrics.classification_report(y_test, predict_test))
print('\n')
print('Confusion Matrix \n',metrics.confusion_matrix(y_test, predict_test))
print('\n')
print('Accuracy of Test : {0:0.3f}'.format(metrics.accuracy_score(y_test, predict_test)))

heat_map = sns.heatmap(
    data = pd.DataFrame(confusion_matrix(y_train,predict_train)),
    annot = True,
    fmt = 'd',
    cmap = sns.color_palette('Blues',50),
    )

#------------------------------------TFIDF COM NAIVE BAYES-----------------------------------------------------------------#

tfidf_transform = TfidfVectorizer().fit_transform(base_treino['text'])

data_count_tfidf = np.array(tfidf_transform.todense(), dtype='float32')

x_train , x_val , y_train , y_val = train_test_split(
    data_count_tfidf , base_treino['label'], test_size = 0.2, random_state = 0)

naive = MultinomialNB()
modelo = naive.fit(x_train,y_train)

predict_train = modelo.predict(x_train)

print('\nClassification Reporte - Tfidf com Naive Bayes \n',metrics.classification_report(y_train, predict_train))
print('\n')
print('Confusion Matrix \n',metrics.confusion_matrix(y_train, predict_train))
print('\n')
print('Accuracy of train : {0:0.3f}'.format(metrics.accuracy_score(y_train,predict_train)))

heat_map = sns.heatmap(
    data = pd.DataFrame(confusion_matrix(y_train,predict_train)),
    annot = True,
    fmt = 'd',
    cmap = sns.color_palette('Blues',50)
    
    )



#-----------------------------------------TFIDF COM RANDOM FOREST-------------------------------------------------#

ramdomforest = RandomForestClassifier()
modelo = ramdomforest.fit(x_train,y_train)

predict_train = modelo.predict(x_train)
predict_test = modelo.predict(x_test)

print('\nClassification Reporte Train - TFIDF com Random Forest\n',metrics.classification_report(y_train, predict_train))
print('\n')
print('Confusion Matrix \n',metrics.confusion_matrix(y_train, predict_train))
print('\n')
print('Accuracy of train : {0:0.3f}'.format(metrics.accuracy_score(y_train,predict_train)))

print('\nClassification Reporte Test - TFIDF com Random Forest\n',metrics.classification_report(y_test, predict_test))
print('\n')
print('Confusion Matrix \n',metrics.confusion_matrix(y_test, predict_test))
print('\n')
print('Accuracy of train : {0:0.3f}'.format(metrics.accuracy_score(y_test, predict_test)))

heat_map = sns.heatmap(
    data = pd.DataFrame(confusion_matrix(y_train,predict_train)),
    annot = True,
    fmt = 'd',
    cmap = sns.color_palette('Blues',50),
 
#----------------------------------------------TESTAR UMA FRASE-------------------------------------------------#
frase = 'muito ruim'
frase_teste = TfidfVectorizer().fit_transform([frase])
data_frase_teste = np.array(frase_teste.todense(), dtype='float32')
predicao = modelo.predict(frase_teste)
print(f'Classe prevista: {predicao}')










