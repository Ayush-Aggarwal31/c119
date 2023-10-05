#Text Data Preprocessing Lib
import nltk
nltk.download('punkt')
from nltk.stem import PorterStemmer
stemmer = PorterStemmer() # root words

import json 
import pickle
import numpy as np

words=[]
classes = [] # greeting and goodbye
word_tags_list = [] # (Hello ,  greeting)
ignore_words = ['?', '!',',','.', "'s", "'m"]
train_data_file = open('intents.json').read()
intents = json.loads(train_data_file)

# function for appending stem words
def get_stemWord(words ,ignore_words ):
    stem_words =[]
    for word in words:
        if word not in ignore_words:
            w = stemmer.stem(word.lower())
            stem_words.append(w)
    return stem_words

    
for intent in intents["intents"]:
    
        # Add all words of patterns to list
        for pattern in intent["patterns"]:
             pattern_word = nltk.word_tokenize(pattern)
             words.extend(pattern_word)
             word_tags_list.append((pattern_word,intent["tag"]))
             
        # Add all tags to the classes list
        if intent["tag"] not in classes:
             classes.append(intent["tag"])
             stem_words = get_stemWord(words,ignore_words)         

#Create word corpus for chatbot
def createbotcorpus(stemwords,classes):
     stemwords=sorted(list(set(stemwords)))
     classes=sorted(list(set(classes)))
     pickle.dump(stemwords,open("words.pkl","wb"))
     pickle.dump(classes,open("classes.pkl","wb"))
     return stemwords,classes
stem_words,classes=createbotcorpus(stem_words,classes)

print(classes)
print(stem_words)

#print(stem_word)
#print(word_tags_list[0])
print(classes)

training_data = []
number_of_tags = len(classes)
labels = [0]*number_of_tags

for wordtags in word_tags_list:
     bagofwords=[]
     pattern_words=wordtags[0]
     for word in pattern_words:
        index=pattern_words.index(word)
        word=stemmer.stem(word.lower())
        pattern_words[index]=word
     for word in stem_words:
        if word in pattern_words:
                bagofwords.append(1)
        else:
             bagofwords.append(0)
     print(bagofwords)

     labels_encoding = list(labels) #labels all zeroes initially
     tag = wordtags[1] #save tag
     tag_index = classes.index(tag)  #go to index of tag
     labels_encoding[tag_index] = 1  #append 1 at that index
       
     training_data.append([bagofwords, labels_encoding])
        # [ 0 , 0, ,0,1,0,0,1] , [0,1,0,0]
print(training_data[0])

def preprocess_train_data(training_data):
   
    training_data = np.array(training_data, dtype=object)
    
    train_x = list(training_data[:,0]) # sentence of stem words
    train_y = list(training_data[:,1]) # tags

    print(train_x[0])
    print(train_y[0])
  
    return train_x, train_y

train_x, train_y = preprocess_train_data(training_data)