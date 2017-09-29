from utilities import *
import numpy as np
from tqdm import tqdm
import string
import nltk
import os
import re
import cPickle as pickle
import random
from string import punctuation


train_sentences = []
test_sentences = []

words = []


def strip_punctuation(s):
    return s.translate(string.maketrans("",""), string.punctuation)
    # return ''.join(c for c in s if c not in punctuation)

def process_text(string):
    #x = re.sub('[^A-Za-z0-9]+', ' ', x)
    string = string.lower()
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)     
    string = re.sub(r"\'s", " \'s", string) 
    string = re.sub(r"\'ve", " \'ve", string) 
    string = re.sub(r"n\'t", " n\'t", string) 
    string = re.sub(r"\'re", " \'re", string) 
    string = re.sub(r"\'d", " \'d", string) 
    string = re.sub(r"\'ll", " \'ll", string) 
    string = re.sub(r",", " , ", string) 
    string = re.sub(r"!", " ! ", string) 
    string = re.sub(r"\(", " \( ", string) 
    string = re.sub(r"\)", " \) ", string) 
    string = re.sub(r"\?", " \? ", string) 
    string = re.sub(r"\s{2,}", " ", string)    
    string = string.split(' ')
    string = [strip_punctuation(s) for s in string]
    # ptxt = nltk.word_tokenize(ptxt)
    return string


if __name__ == "__main__":
    file = open("../data/rt-polarity.pos", "r")
    for line in file:
        s = {}
        ptxt = process_text(line)
        s["text"] = ptxt
        s["polarity"] = "pos"
        words += ptxt
        train_sentences.append(s)
    file.close()

    file = open("../data/rt-polarity.neg", "r")
    for line in file:
        s = {}
        ptxt = process_text(line)
        s["text"] = ptxt
        s["polarity"] = "neg"
        words += ptxt
        train_sentences.append(s)
    file.close()

    file = open("../data/test_file_pos", "r")
    for line in file:
        s = {}
        ptxt = process_text(line)
        s["text"] = ptxt
        s["polarity"] = "pos"
        words += ptxt
        test_sentences.append(s)
    file.close()

    file = open("../data/test_file_neg", "r")
    for line in file:
        s = {}
        ptxt = process_text(line)
        s["text"] = ptxt
        s["polarity"] = "neg"
        words += ptxt
        test_sentences.append(s)
    file.close()

    words = list(set(words))

    print("{} unique words".format(len(words)))
    print("{} train sentences".format(len(train_sentences)))
    print("{} test sentences".format(len(test_sentences)))

    # Building vocab indices
    index_word = {index+2:word for index,word in enumerate(words)}
    word_index = {word:index+2 for index,word in enumerate(words)}
    index_word[0], index_word[1] = '<pad>','<unk>'
    word_index['<pad>'], word_index['unk'] = 0,1


    # Avoid using 0 incase we want to pad zero vectors
    sentiment_map = {
        'pos':1,
        'neg':0
    }

    advertising_lexicon = advertising_lexicon_load("../dic/effective_advertising_phrases")

    training_set, testing_set = [], []
    for sent in train_sentences:
        txt = sent['text']
        tokenized_txt = [word_index[x] for x in txt]
        actual_len = len(tokenized_txt)
        advertising_words = []
        for x in txt:
            if x in advertising_lexicon:
                advertising_words.append(x)
        tmp = {"tokenized_txt": tokenized_txt, "actual_len": actual_len, "advertising_words": advertising_words, "polarity": sentiment_map[sent["polarity"]]}
        training_set.append(tmp)

    for sent in test_sentences:
        txt = sent['text']
        tokenized_txt = [word_index[x] for x in txt]
        actual_len = len(tokenized_txt)
        advertising_words = []
        for x in txt:
            if x in advertising_lexicon:
                advertising_words.append(x)
        tmp = {"tokenized_txt": tokenized_txt, "actual_len": actual_len, "advertising_words": advertising_words, "polarity": sentiment_map[sent["polarity"]]}
        testing_set.append(tmp)

    print("Spliting into Dev Set")
    random.shuffle(training_set)
    dev_set = training_set[:500]
    training_set = training_set[500:]

    env = {
    "index_word":index_word,
    "word_index":word_index,
    "train":training_set,
    "dev":dev_set,
    "test":testing_set,
    #'max_len':max_len
    }

    
    glove = {}
    dimensions = 300

    glove_path = "glove_embeddings.pkl"

    with open('../../../../NLP_experiment/tools/glove.6B/glove.6B.{}d.txt'.format(dimensions),'r') as f:
        lines = f.readlines()
        for l in tqdm(lines):
            vec = l.split(' ')
            word = vec[0].lower()
            vec = vec[1:]
            #print(word)
            #print(len(vec))
            glove[word] = np.array(vec)

    print('glove size={}'.format(len(glove)))
    save = True
    
    print("Finished making glove dictionary")

    matrix = np.zeros((len(word_index), dimensions))
    #print(matrix.shape)

    oov = 0 

    filtered_glove = {}
    for i in tqdm(range(2, len(word_index))):
        word = index_word[i]
        if(word in glove):
            vec = glove[word]
            if(save==True):
                filtered_glove[word] = glove[word]
            # print(vec.shape)
            #matrix = np.vstack((matrix,vec))
            matrix[i] = vec
        else:
            random_init = np.random.uniform(low=-0.01,high=0.01, size=(1,dimensions))
            # print(random_init)
            #matrix = np.vstack((matrix,random_init))
            matrix[i] = random_init
            oov +=1
            # print(word)

    if(save==True):
        with open(glove_path,'w') as f:
            pickle.dump(filtered_glove, f)
        print("Saving glove dict to file")
    

    print(matrix.shape)
    print(len(word_index))
    print("oov={}".format(oov))

    print("Saving glove vectors")
    env['glove'] = matrix

    file_path = "env.pkl"
    with open(file_path, "w") as f:
        pickle.dump(env, f)
