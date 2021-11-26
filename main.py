import json
from math import pi
import pickle
import colorama
colorama.init()
from colorama import Fore,Style,Back
from numpy.lib.npyio import load
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Embedding,GlobalAveragePooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from tensorflow.python.ops.gen_math_ops import Max

with open('intents.json') as file:
    data=json.load(file)
training_sentences=[]
training_labels=[]
labels=[]
responses=[]
for intent in data['intents']:
    for pattern in intent['patterns']:
        training_sentences.append(pattern)
        training_labels.append(intent['tag'])
    responses.append(intent['responses'])
    if intent['tag'] not in labels:
        labels.append(intent['tag'])
num_labels=len(labels)

lbl_encoder=LabelEncoder()
lbl_encoder.fit(training_labels)
training_labels=lbl_encoder.transform(training_labels)

vocab_size=1000
embedding_dim=16
max_len=20
oov_token="<OOV>"

tokenizer=Tokenizer(num_words=vocab_size,oov_token=oov_token)
tokenizer.fit_on_texts(training_sentences)
word_index=tokenizer.word_index
sequences=tokenizer.texts_to_sequences(training_sentences)
padded_sequences=pad_sequences(sequences,truncating='post',maxlen=max_len)

model=Sequential()
model.add(Embedding(vocab_size,embedding_dim,input_length=max_len))
model.add(GlobalAveragePooling1D())
model.add(Dense(16,activation='relu'))
model.add(Dense(16,activation='relu'))
model.add(Dense(num_labels,activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()

epochs=500
history=model.fit(padded_sequences,np.array(training_labels),epochs=epochs)
model.save('chat_model')

with open('tokenizer.pickle','wb') as handle:
    pickle.dump(tokenizer,handle,protocol=pickle.HIGHEST_PROTOCOL)

with open('lbl_encoder.pickle','wb') as enc_file:
    pickle.dump(lbl_encoder,enc_file,protocol=pickle.HIGHEST_PROTOCOL)

def chat():
    model=keras.models.load_model('chat_model')

    with open('tokenizer.pickle','rb') as handle:
        tokenizer=pickle.load(handle)
     
    with open('lbl_encoder.pickle','rb') as enc:
        lbl_encoder=pickle.load(enc)
    
    max_len=20

    while True:
        print(Fore.LIGHTBLUE_EX + "User: " + Style.RESET_ALL,end="")
        inp=input()
        if inp.lower()=="quit":
            break
        result=model.predict(keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences([inp]),truncating='post',maxlen=max_len))
        tag=lbl_encoder.inverse_transform([np.argmax(result)])
        for i in data['intents']:
            if i['tag']==tag:
                print(Fore.GREEN + "Chatbot:" + Style.RESET_ALL,np.random.choice(i['responses']))
print(Fore.YELLOW + "Start Messaging with bot (type quit to stop)!"+Style.RESET_ALL)
chat()