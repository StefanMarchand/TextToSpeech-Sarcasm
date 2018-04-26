#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 12:56:27 2018

@author: Luis Sanchez, Jonathan

Purpose: Retrive the tensorflow model, and use it to translate unknown words into phonetic
         If the is known in the English dictionary, they use regex to search through them
         and return the phonetic used.
"""
import tensorflow as tf
import pickle
import re

# must be the same as used to built model using tensorflow
batch_size = 64

cmu_dict = open('Dataset_v1/cmudict.dict', "r")

# conv the dict into a string to regex over it
dictString = cmu_dict.read()
# close dictionary
cmu_dict.close()

def wordsToPhonetics(test):
    loadPickle()
    # parse through text converting to arpabet
    seg_test = test.split(" ")
    # contain content for sentence given to phonetics
    wordsToPhonetic = []

    
    # For each word in the given sentence find its phonetic spelling
    for word in seg_test:
        # word to regex
        wordRex =  '\\n' + word.lower() + '\s(.*)'
        p = re.compile(wordRex)
        # search in dictionary
        matchedObj = re.search(p, dictString)
        if (matchedObj == None):
            print("don't know it:", word.lower())
            guess = translate_sentence(word)
            wordsToPhonetic.append(guess)
            continue
        # add phonetic to dictionary
        print(matchedObj.group(1))
        wordsToPhonetic.append(matchedObj.group(1))

    print(wordsToPhonetic)
    # return value
    return wordsToPhonetic

def loadPickle():
    global load_path
    global source_vocab_to_int
    global target_vocab_to_int
    global source_int_to_vocab
    global target_int_to_vocab


    save_path = 'checkpoints/dev'
    with open('preprocess.p', mode='rb') as in_file:
        LOADED = pickle.load(in_file)
    _, (source_vocab_to_int, target_vocab_to_int), (source_int_to_vocab, target_int_to_vocab) = LOADED
    load_path = pickle.load(open('params.p', mode='rb'))
    print(source_vocab_to_int, target_vocab_to_int, source_int_to_vocab, target_int_to_vocab)




#XXXX MODIFIED FOR OUR PROBLEM XXXX
def word_to_seq(word, vocab_to_int):
    """
    Convert a word to a sequence of ids
    :param sentence: String
    :param vocab_to_int: Dictionary to go from the words to an id
    :return: List of word ids
    """
    # TODO: Implement Function
    '''Prepare the text for the model'''
    seq=[]
    for char in word:
        if char not in vocab_to_int:
            seq.append(vocab_to_int['<UNK>'])
        else:
            seq.append(vocab_to_int[char])
    return seq

def translate_word(word):
    global source_vocab_to_int
    word_seq = word_to_seq(word.lower(), source_vocab_to_int)

    loaded_graph = tf.Graph()
    with tf.Session(graph=loaded_graph) as sess:
        # Load saved model
        loader = tf.train.import_meta_graph(load_path + '.meta')
        loader.restore(sess, load_path)

        input_data = loaded_graph.get_tensor_by_name('input:0')
        logits = loaded_graph.get_tensor_by_name('predictions:0')
        target_sequence_length = loaded_graph.get_tensor_by_name('target_sequence_length:0')
        source_sequence_length = loaded_graph.get_tensor_by_name('source_sequence_length:0')
        keep_prob = loaded_graph.get_tensor_by_name('keep_prob:0')

        translate_logits = sess.run(logits, {input_data: [word_seq]*batch_size,
                                             target_sequence_length: [len(word_seq)*2]*batch_size,
                                             source_sequence_length: [len(word_seq)]*batch_size,
                                             keep_prob: 1.0})[0]


    return  '{}'.format(" ".join([target_int_to_vocab[i] for i in translate_logits]))


def translate_sentence(sentence):
    sentence_seq = ""
    sentence = sentence.split(" ")
    for word in sentence:
        sentence_seq += translate_word(word)[:-5]
    return sentence_seq


def sentence_to_seq(sentence, vocab_to_int):
    global source_int_to_vocab
    global target_int_to_vocab
    """
    Convert a sentence to a sequence of ids
    :param sentence: String
    :param vocab_to_int: Dictionary to go from the words to an id
    :return: List of word ids
    """
    return [vocab_to_int.get(word, vocab_to_int['<UNK>']) for word in sentence.lower().split()]


def translateUnk(word):
    print(word)
    global source_vocab_to_int
    global translate_sentence
    translate_sentence = sentence_to_seq(word, source_vocab_to_int)

    loaded_graph = tf.Graph()
    with tf.Session(graph=loaded_graph) as sess:
        # Load saved model
        loader = tf.train.import_meta_graph(load_path + '.meta')
        loader.restore(sess, load_path)

        input_data = loaded_graph.get_tensor_by_name('input:0')
        logits = loaded_graph.get_tensor_by_name('predictions:0')
        target_sequence_length = loaded_graph.get_tensor_by_name('target_sequence_length:0')
        source_sequence_length = loaded_graph.get_tensor_by_name('source_sequence_length:0')
        keep_prob = loaded_graph.get_tensor_by_name('keep_prob:0')

        translate_logits = sess.run(logits, {input_data: [translate_sentence]*batch_size,
                                             target_sequence_length: [len(translate_sentence)*2]*batch_size,
                                             source_sequence_length: [len(translate_sentence)]*batch_size,
                                             keep_prob: 1.0})[0]

    print('Input')
    print('  Word Ids:      {}'.format([i for i in translate_sentence]))
    print('  English Word: {}'.format([source_int_to_vocab[i] for i in translate_sentence]))

    print('\nPrediction')
    print('  Word Ids:      {}'.format([i for i in translate_logits]))
    print('  Phones: {}'.format(" ".join([target_int_to_vocab[i] for i in translate_logits])))
