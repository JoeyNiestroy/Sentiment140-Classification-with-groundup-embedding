import pandas as pd
import numpy as np
import tensorflow as tf
from multiprocessing import Pool
from functools import partial

"""Code to generate training samples from previously made DF, based off work from Word2Vec"""

"""Function to convert sentences to index based off Dictionary"""
def conver_int(sentence):
    int_array = np.zeros(len(sentence))
    for index in range(0,len(sentence)):
        int_array[index] = vocab_dic[sentence[index]]
    return (int_array).tolist()

"""Function to genrate positive samples using index encoded sentences, no negative sampling, all labels are dropped"""
def positive_sample_gen(econoded_array):
    count = False
    econoded_array = [econoded_array]
    for sentence in econoded_array:
        data = tf.keras.preprocessing.sequence.skipgrams(sentence, vocabulary_size = 46971, window_size = 3, negative_samples = 0)
        arr = np.asarray(data[0])
        if count and len(arr)>0:
            train_data_positive = np.concatenate((train_data_positive,arr), axis = 0)
        if count is False:
            train_data_positive = arr
            count = True
    return train_data_positive

def negative_sample_gen(prob_array, postive_sample):
    max_int = int(postive_sample[:,0].max())
    main_neg_sample_array = []
    for index in range(1,max_int+1):
        index_array = postive_sample[postive_sample[:,0]== index]
        num_neg_samples = 6*len(index_array)
        neg_sample_array = []
        neg_samples = np.random.choice(prob_array, round(4*(num_neg_samples)))
        """No condictional is used to resample if 4x is not enough it is gods will there are less than 5x neg samples"""
        for sample in neg_samples:
            if len(neg_sample_array) >= num_neg_samples:
                break
            else:
                if sample in index_array[:,1] or sample == index:
                    pass
                else:
                    arr = np.array([index,sample])
                    neg_sample_array.append(arr)

    main_neg_sample_array.extend(neg_sample_array)
    train_data_negative = (np.array(main_neg_sample_array))
    return train_data_negative
if __name__ == "__main__":
    """Main Code Block: Many items are hard-coded in as code was pulled from Juypter and fitted to .py file for purpose of 
    multiprocessing and use on HPC system"""
    
    
    """Reads in Final Dataframe from preprocessing and creates array of all tweets"""
    df = pd.read_csv("Final_Data.csv")
    final_array = []
    for sentence in df["Prep_3"]:
        array = eval(sentence)
        final_array.extend(array)

    """Creating integer dic for words as well as dictionary for counts of words for later prob matrix"""
    vocab_dic, index = {}, 1  
    word_count = {}
    vocab_dic['<pad>'] = 0  
    for word in final_array:
        if word not in vocab_dic:
            vocab_dic[word] = index
            index += 1
        if word not in word_count:
            word_count[word] = 1
        else:
            word_count[word] += 1
    vocab_size = len(vocab_dic)

    """Creates List of converted sentences, no memory or speed issues using python list here"""
    encoded_sent = []
    for sentence in df["Prep_3"]:
        encoded_sent.append(conver_int(eval(sentence)))

    """Calculations to determine sum of probabilties to build unigram array (See Word2Vec paper 2)"""
    sum_1 = 0
    for word in word_count.keys():
        prob = ((word_count[word])/17210107)**(3/4)
        sum_1 += prob

    """Code block to build array of len 100M which will be used to generate neagtive samples"""
    prob_table = np.zeros(100000000)
    curr = 0
    for word in word_count.keys():
        prob = (word_count[word])**(3/4)
        prob = prob/267201
        prob = prob/sum_1
        num_values = round(prob*100000000)
        prob_table[curr:(curr+num_values)] = vocab_dic[word]
        curr = curr+num_values
    unigram_array = np.trim_zeros(prob_table)
    
    """Sets up args for functions in MP"""
    function = partial(negative_sample_gen, unigram_array)

    """Functions run on MP"""
    with Pool(12) as p:
        train_data_positive = p.map(positive_sample_gen, encoded_sent)
    with Pool(12) as p:
        train_data_negative = p.map(function,train_data_positive )
    
    """Minor data alterations"""
    final_pos = []
    for list in train_data_positive:
        final_pos.extend(list)
    final_neg = []
    for list in train_data_negative:
        final_neg.extend(list)
    final_pos = (np.array(final_pos))
    final_neg = (np.array(final_neg))
    print(len(final_pos))
    print(len(final_neg))
    #np.save('Train_data_positive_Test.npy',final_pos)
    #np.save('Train_data_negative_Test.npy',final_neg)
