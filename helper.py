import numpy as np

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

