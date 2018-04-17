import re
import itertools
import codecs
from collections import Counter
import pickle
import os
import numpy as np
import gensim
import matplotlib.pyplot as plt
maxLen = 600
trainPath = '/media/SSD/LinuxData/DataSet/aclImdb/train'
testPath = '/media/SSD/LinuxData/DataSet/aclImdb/test'
vocabPath = '/media/SSD/LinuxData/DataSet/aclImdb/imdb.vocab'
def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    print('data clean.........')
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
    return string.strip().lower()

def load_data(dataPath=None):
    if dataPath == None:
        print('no dataPath to load......')
        exit()
    print('loading text data.....')
    print('loading pos data.....')
    posPath = os.path.join(dataPath,'pos')
    fileList = os.listdir(posPath)
    positive_example = []
    positive_label = []
    for filename in fileList:
        line = open(os.path.join(posPath,filename)).readline()
        positive_example.append(line)
        positive_label.append([1,0])

    #print(np.shape(positive_example))
    print('pos data load finish ! shape: ',np.shape(positive_example))
    print('loading neg data......')
    negPath = os.path.join(dataPath,'neg')
    fileList = os.listdir(negPath)
    neg_example = []
    neg_label = []
    for filename in fileList:
        line = open(os.path.join(negPath,filename)).readline()
        neg_example.append(line)
        neg_label.append([0,1])

    print('neg data load finish ! shape: ',np.shape(neg_example))
    x_text = positive_example + neg_example
    x_text = [clean_str(strs).split() for strs in x_text]
    print('data clean finished !')
    '''
    numWords = []
    for line in x_text:
        numWords.append(len(line))
    print('file num: ',len(numWords),'\ntotal words: ',sum(numWords),'\naverage words: ',sum(numWords)/len(numWords))
    print('max len: ',max(numWords))
    plt.hist(numWords,50)
    plt.xlabel('sequence length')
    plt.ylabel('frequency')
    plt.axis([0,1200,0,8000])
    plt.show()
    '''

    #x_label = positive_label + neg_label
    x_label = np.concatenate([positive_label, neg_label], 0)

    #print('shape: ',np.shape(x_text),np.shape(x_label),x_text[0])
    print('loading data success .')
    return [x_text,x_label]

#load_data(trainPath)

def build_vocab():

    with open(vocabPath) as f:
        vocab_dict = {word:i+1 for i,word in enumerate(f.readlines())}
    vocab_size = len(vocab_dict)
    return vocab_dict,vocab_size

def pad_sentence(sentences,maxLen=maxLen,pading='00'):
    print('padding sentence..........')
    padded_sentence = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        num_pad = maxLen - len(sentence)
        if num_pad >= 0:
            new_sentence = sentence + [pading]*num_pad
            padded_sentence.append(new_sentence)
        else:
            new_sentence = sentence[:maxLen]
            padded_sentence.append(new_sentence)
    #print('padding shape: ',np.shape(np.array(padded_sentence)))
    print('padding sentence success.')
    print('after padding,the shape of x_text is: ',np.shape(padded_sentence))
    print('x_text: ')
    #for i in range(10):
    #    print(padded_sentence[i])
    return padded_sentence

def build_input_data(data,lables,wordvecPath,batch_size,num_epoches,vector_size):
    data = np.array(data)
    lables = np.array(lables)
    data_size = len(data)
    num_batches_per_epoch = int(data_size / batch_size) + 1
    #max_length = max(len(x) for x in data)
    print('loading google word2vec....................')
    model = gensim.models.KeyedVectors.load_word2vec_format(wordvecPath,binary=True)
    print('load google word2vec success !')
    #vocab_dict,vocab_size = build_vocab()
    W = []
    f = open('wordvec.pkl','wb')
    input_data = []
    for epoch in range(num_epoches):
        print('writting epoch: ',epoch,'/',num_epoches,'  .........')
        shuffle_indices = np.random.permutation(np.arange(data_size))
        print('shuffling data....................')
        shuffle_data = data[shuffle_indices]
        shuffle_label = lables[shuffle_indices]
        print('shuffling data finished .')
        for batch_num in range(num_batches_per_epoch):
            print('writing batch ',batch_num,'  at epoch ',epoch,' .............')
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            batch_data = shuffle_data[start_index:end_index]
            Sentence_vec = []
            for single_sentence in batch_data:
                sin_sen_vec = []
                for word in single_sentence:
                    try:
                        vec = model[word]
                    except:
                        vec = np.zeros(shape=(vector_size),dtype=np.float32)
                    sin_sen_vec.append(vec)
                Sentence_vec.append(sin_sen_vec)
            input_data.append([Sentence_vec,shuffle_label[start_index:end_index]])
            print('batch num: ',len(input_data))
            pickle.dump([Sentence_vec,shuffle_label[start_index:end_index]],f)
            print('batch ',batch_num,' at epoch ',epoch,' writing finished .')

        print('epoch ',epoch,' writing finished .')

    #pickle.dump(W,open('wordvec.pkl','wb'))
    return input_data


wordvecPath = '/media/SSD/LinuxData/model/goole_word2vec/GoogleNews-vectors-negative300.bin.gz'
x_text,x_label = load_data(trainPath)
x_text = pad_sentence(x_text)
#build_input_data(np.array(x_text),np.array(x_label),wordvecPath,25,10,300)
build_input_data(x_text,x_label,wordvecPath,100,5,300)
