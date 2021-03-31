import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
import torch as torch
from torch import nn
from nltk.tokenize import word_tokenize
import math as mt
import time
import random
from joblib import Parallel, delayed
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch.utils.data as data_utils
from torch.utils.data import TensorDataset, DataLoader
from sklearn.utils import shuffle
from models.LSTM import SentimentLSTM
from models.LSTM import SentimentLSTM_task3
from models.LSTM import SentimentLSTM_bengali
from models.LSTM import SentimentLSTM_task3_bengali
from models.Word2Vec import Word2Vec


def apply_stopword_removal(data):
    hindi_stopword_file = open('data/stopwords-hi.txt', encoding="utf8")

    sw_list = ['#', '?', '!', ';', ',', ':', "\'", '-', '=', '(', ')', '[', ']', '{', '}', '"', '*', '@', '  ', '\\',
               '/', '..', '...', '....', '%'
        , '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '\t']
    sw_list_string = ''
    for i in sw_list:
        sw_list_string += i
    hindi_stopwords = []
    for x in hindi_stopword_file:
        hindi_stopwords.append(x.rstrip())

    hindi_stopwords.extend(sw_list)
    sentences = []
    for text in data['text']:
        text_array = text.split(' ')
        new_array = []
        for j in text_array:
            if '@' not in j and len(j) < 20:
                for char in sw_list:
                    j = j.replace(char, '')
                new_array.append(j.lower())
        sentences.append(' '.join(new_array))
    return sentences

def apply_stopword_removal_task3(data):
    hindi_stopword_file = open('data/stopwords-hi.txt', encoding="utf8")

    sw_list = ['#', '?', '!', ';', ',', ':', "\'", '-', '=', '(', ')', '[', ']', '{', '}', '"', '*', '@', '  ', '\\',
               '/', '..', '...', '....', '%'
        , '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '\t']
    sw_list_string = ''
    for i in sw_list:
        sw_list_string += i
    hindi_stopwords = []
    for x in hindi_stopword_file:
        hindi_stopwords.append(x.rstrip())

    hindi_stopwords.extend(sw_list)
    sentences = []
    for text in data['text']:
        text_array = text.split(' ')
        new_array = []
        for j in text_array:
            if '@' not in j and len(j) < 20 and 'www.' not in j and '.com' not in j and 'http' not in j:
                for char in sw_list:
                    j = j.replace(char, '')
                new_array.append(j.lower())
        sentences.append(' '.join(new_array))
    return sentences


def build_vocabulary(sentences):
    temp_unique = []  # For unique words
    temp_nounique = []
    for j in sentences:
        temp2 = j.split(' ')
        for k in temp2:
            if k not in temp_unique:
                temp_unique.append(k)
            temp_nounique.append(k)
    return temp_unique, temp_nounique

def word_to_one_hot(word, features):
    if word in V:
        index = V.index(word)
        encoding = np.zeros(features)
        encoding[index]= 1
        return encoding.astype(np.uint8)
    return False

def not_word_to_one_hot(word, features):
    if word in V:
        index = V.index(word)
        return index
    return False

def index_to_onehot(X_batch):
    X_batch_new = []
    y_batch_new = []
    encodingX = np.eye(4)[X_batch]
    return type(encodingX)

def sentence_to_index(sentence, V):
    words = sentence.split(' ')
    l = []
    for i in words:
        l.append(V.index(i) + 1)
    return l, len(l)
#     return l

def padding(array, seq_len):
    padded_array = []
    for item in array:
        if len(item) < seq_len:
            padded_array.append([0]*(seq_len-len(item)) + item)
        else:
            padded_array.append(item)
    return np.array(padded_array)

def sentence_to_numeric_arr(sentences, V):
    x_data = []
    max_len_curr = -1
    for sentence in sentences:
        temp, max_len = sentence_to_index(sentence, V)
        if max_len > max_len_curr:
            max_len_curr = max_len
        x_data.append(temp)
    return x_data, max_len_curr

def get_bengali_data(file_path):
    bengali_data = pd.read_csv(file_path)

    bengali_data_hate = bengali_data.loc[bengali_data['hate'] == 1]
    bengali_data_not_hate = bengali_data.loc[bengali_data['hate'] == 0]

    bengali_data_hate = bengali_data_hate.iloc[0:2332] 
    bengali_data_not_hate = bengali_data_not_hate.iloc[0:2333]

    bengali_data = pd.concat([bengali_data_hate, bengali_data_not_hate])
    bengali_data = shuffle(bengali_data)
    bengali_data.columns = ["text", "hate", "category"]

    labels = bengali_data['hate']
    labels = np.array(labels)
    
    return bengali_data, labels

def load_word2vec_embeddings(model_path, device, features, embedding_size):
    model = Word2Vec(features, embedding_size)
    if(device == "cpu"):
        model.cpu()
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    else:
        model.load_state_dict(torch.load(model_path))

    print(model.eval())
    weights1 = torch.transpose(model.fc1.weight, 0, 1)
    weights2 = torch.transpose(model.fc2.weight, 0, 1)
    temp_row = np.zeros(embedding_size).reshape(1,embedding_size)
    weights1_np = weights1.cpu().detach().numpy()
    weights1_np = np.concatenate((temp_row, weights1_np), axis=0)
    weights1 = torch.Tensor(weights1_np)
    
    return weights1, weights2


def split_data_train_valid_test(data_x, labels_y, batch_size):
    train_ratio = 0.8
    valid_ratio = (1 - train_ratio)/2
    total = data_x.shape[0]
    train_cutoff = int(total * train_ratio)
    valid_cutoff = int(total * (1 - valid_ratio))

    train_x, train_y = data_x[:train_cutoff], labels_y[:train_cutoff]
#     valid_x, valid_y = data_x[train_cutoff : valid_cutoff], labels_y[train_cutoff : valid_cutoff]
    test_x, test_y = data_x[train_cutoff:], labels_y[train_cutoff:]
#     print(type(train_x), type(train_y))
    temp = torch.tensor(train_x)
    train_data = TensorDataset(torch.tensor(train_x), torch.tensor(train_y))
#     valid_data = TensorDataset(torch.tensor(valid_x), torch.tensor(valid_y))
    test_data = TensorDataset(torch.tensor(test_x), torch.tensor(test_y))

    train_loader = DataLoader(train_data, batch_size = batch_size, shuffle = True, drop_last=True)
#     valid_loader = DataLoader(valid_data, batch_size = batch_size, shuffle = True, drop_last=True)
    test_loader = DataLoader(test_data, batch_size = batch_size, shuffle = True, drop_last=True)
    
    return train_loader, test_loader

def initialize_SentimentLSTM_model(n_vocab, n_embed, n_hidden, n_output, n_layers, device, embedding_weights):
    net = SentimentLSTM(n_vocab, n_embed, n_hidden, n_output, n_layers, embedding_weights)
    net.to(device)
    criterion = nn.BCELoss()
    
    return net, criterion

def initialize_SentimentLSTM_model_bengali(n_vocab, n_embed, n_hidden, n_output, n_layers, device, embedding_weights):
    net = SentimentLSTM_bengali(n_vocab, n_embed, n_hidden, n_output, n_layers, embedding_weights)
    net.to(device)
    criterion = nn.BCELoss()
    
    return net, criterion

def initialize_SentimentLSTM_model_task3(n_vocab, batch_size, n_embed, n_hidden, n_output, n_layers, device):
    net = SentimentLSTM_task3(n_vocab, batch_size, n_embed, n_hidden, n_output, n_layers)
    net.to(device)
    criterion = nn.BCELoss()
    
    return net, criterion

def initialize_SentimentLSTM_model_task3_bengali(n_vocab, batch_size, n_embed, n_hidden, n_output, n_layers, device, embedding_weights):
    net = SentimentLSTM_task3_bengali(n_vocab, batch_size, n_embed, n_hidden, n_output, n_layers, embedding_weights)
    net.to(device)
    criterion = nn.BCELoss()
    return net, criterion
       

def sampling_prob(word, total_words): #non_unique is the count of total words
    z_wi = total_words.count(word) / len(total_words)
    try:
        p_wi = (mt.sqrt(z_wi / 0.001) + 1) * (0.001 / z_wi)
        return p_wi
    except ZeroDivisionError:
        print("Word doesn't exist in corpus")
    pass

def get_target_context(sentence, window_size, total_words):
    word = ''
    words_in_sentence = sentence.split(' ')
    for i in range(len(words_in_sentence)):
        context = []
        randd = random.random()
        if randd <= sampling_prob(words_in_sentence[i], total_words):
            word = words_in_sentence[i]
            upper_bound = i + window_size + 1
            lower_bound = i - window_size
            for j in range(lower_bound, upper_bound):
                rand2 = random.random()
                if i != j and j>=0 and j<len(words_in_sentence):
                    if rand2 <= sampling_prob(words_in_sentence[j], total_words):
                        context.append(words_in_sentence[j])
        if len(word) > 0 and len(context) > 0:
            yield(word, context)

def create_dataset(sentences, total_words, V, window_size): #Creating target, context tuple
    x_train = []
    y_train = []
    count = 0
    print("started")
    for j, i in enumerate(sentences):
        if (j + 1) % 50 == 0:
            print("INDEX:", j, "Length:",len(x_train))
        word_context = get_target_context(i, window_size, total_words)
        for word, context in word_context:
            input_vec = not_word_to_one_hot(word, len(V))
            for j in context:
                output_vec = not_word_to_one_hot(j, len(V))
                x_train.append(input_vec)
                y_train.append(output_vec)
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    np.save('x_train_bengali', x_train)
    np.save('y_train_bengali', y_train)
    return x_train, y_train
