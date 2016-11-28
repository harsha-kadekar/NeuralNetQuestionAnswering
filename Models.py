#######################################################################################################################
# Name: Models.py
# Description: This file has classes and functions which will help you to form the language models. Basically in this file
#               we are constructing our neural networks.
# References: http://smerity.com/articles/2015/keras_qa.html
######################################################################################################################
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers import recurrent
from keras.layers import Dense, Merge, Dropout, RepeatVector
from keras.layers import Convolution1D, MaxPooling1D, Flatten

from ImageDataStructure import ImageData
from DataFileInputOutput import DataFileInputOutput
from preprocessdata import PreprocessingUtilities

import numpy as np


np.random.seed(1337)  # for reproducibility


class StoryQueryLanguageModel(object):
    '''
    This class is used to form the Story query neural networks and build the language models
    '''
    def __init__(self, data_set_size=100, embed_hidden_size=62, rnn=recurrent.LSTM, batch_size=32, epochs=128, phrases_hidden_size=256, question_hiden_size=128, max_phrases_length=15, max_question_length=10, max_answer_length=1, max_number_of_phrases=5, max_features=20000):
        self.DATASET_SIZE = data_set_size
        self.EMBED_HIDDEN_SIZE = embed_hidden_size
        self.RNN = rnn
        self.BATCH_SIZE = batch_size
        self.EPOCHS = epochs
        self.PHRASES_HIDDEN_SIZE = phrases_hidden_size
        self.QUESTION_HIDDEN_SIZE = question_hiden_size
        self.MAX_PHRASE_LENGTH = max_phrases_length
        self.MAX_QUESTION_LENGTH = max_question_length
        self.MAX_ANSWER_LENGTH = max_answer_length
        self.MAX_NUM_OF_PHARSES = max_number_of_phrases
        self.MAX_FEATURES = max_features

        self.word_idx = None
        self.list_image_data = None
        self.vocabulory = None



    def preprocess_the_data(self, type_of_file=-1):
        '''
        This function will read the data from the data sets and then does the preprocessing on it so that data
        is ready to be fed to the neural network
        :param: type_of_file - This will tell which file needs to be read for the data
                By default -1: any user given file
                1 - Only Why questions
                2 - Only What questions
                3 - Only Where questions
                4 - Only Which questions
                5 - Only When questions
                6 - Only Who questions
                7 - Only How questions
                8 or other - random some questions like 100, 1000, 5000, 10000, 50000, all
        :return: -
        '''
        self.list_image_data = None
        if type_of_file == -1:
            # uncomment the line based on the size of data you want to use for training.
            # self.list_image_data = DataFileInputOutput.read_from_sample_file('sample_100.json')    #Data size 100
            # self.list_image_data = DataFileInputOutput.read_from_sample_file('sample_1000.json')   #Data size 1000
            self.list_image_data = DataFileInputOutput.read_from_sample_file('sample_5000.json')   #Data size 5000
            # self.list_image_data = DataFileInputOutput.read_from_sample_file('sample_10000.json')  #Data size 10000
            # self.list_image_data = DataFileInputOutput.read_from_sample_file('sample_50000.json')  #Data size 50000
            # self.list_image_data = DataFileInputOutput.read_from_sample_file('sample_all.json')     #Data size everything
            # self.list_image_data = DataFileInputOutput.read_from_sample_file('sample_why.json')                           #Only why questions
            # self.list_image_data = DataFileInputOutput.read_from_sample_file('sample_what.json', elements_count=50000)    #Only what questions
            # self.list_image_data = DataFileInputOutput.read_from_sample_file('sample_where.json', elements_count=50000)   #Only where questions
            # self.list_image_data = DataFileInputOutput.read_from_sample_file('sample_which.json', elements_count=50000)   #Only which questions
            # self.list_image_data = DataFileInputOutput.read_from_sample_file('sample_when.json', elements_count=50000)    #Only when questions
            # self.list_image_data = DataFileInputOutput.read_from_sample_file('sample_who.json', elements_count=50000)       #Only who questions
            # self.list_image_data = DataFileInputOutput.read_from_sample_file('sample_how.json', elements_count=50000)     #Only how questions
        else:
            if type_of_file == 1:
                print '=================================WHY QUESTION=================================================='
                self.list_image_data = DataFileInputOutput.read_from_sample_file('sample_why.json')
            elif type_of_file == 2:
                print '=================================WHAT QUESTION=================================================='
                self.list_image_data = DataFileInputOutput.read_from_sample_file('sample_what.json',
                                                                                 elements_count=50000)
            elif type_of_file == 3:
                print '=================================WHERE QUESTION=================================================='
                self.list_image_data = DataFileInputOutput.read_from_sample_file('sample_where.json',
                                                                                 elements_count=50000)
            elif type_of_file == 4:
                print '=================================WHICH QUESTION=================================================='
                self.list_image_data = DataFileInputOutput.read_from_sample_file('sample_which.json',
                                                                                 elements_count=50000)
            elif type_of_file == 5:
                print '=================================WHEN QUESTION=================================================='
                self.list_image_data = DataFileInputOutput.read_from_sample_file('sample_when.json',
                                                                                 elements_count=50000)
            elif type_of_file == 6:
                print '=================================WHO QUESTION=================================================='
                self.list_image_data = DataFileInputOutput.read_from_sample_file('sample_who.json',
                                                                                 elements_count=50000)
            elif type_of_file == 7:
                print '=================================HOW QUESTION=================================================='
                self.list_image_data = DataFileInputOutput.read_from_sample_file('sample_how.json',
                                                                                 elements_count=50000)
            else:
                print '=================================SAMPLE QUESTION=================================================='
                # self.list_image_data = DataFileInputOutput.read_from_sample_file('sample_100.json')    #Data size 100
                # self.list_image_data = DataFileInputOutput.read_from_sample_file('sample_1000.json')   #Data size 1000
                # self.list_image_data = DataFileInputOutput.read_from_sample_file('sample_5000.json')   #Data size 5000
                # self.list_image_data = DataFileInputOutput.read_from_sample_file('sample_10000.json')  #Data size 10000
                self.list_image_data = DataFileInputOutput.read_from_sample_file('sample_50000.json')  #Data size 50000
                # self.list_image_data = DataFileInputOutput.read_from_sample_file('sample_all.json')     #Data size everything

        # Get the vocabulory of the data set.
        self.vocabulory = PreprocessingUtilities.get_vocabulary_from_image_data_list(self.list_image_data)
        #Get the word to index used for encoding
        self.word_idx = PreprocessingUtilities.get_word_to_index(self.vocabulory)
        #Get the maximum length of the phrases, questions and answers
        len_comp = PreprocessingUtilities.get_max_length_components(self.list_image_data)
        self.MAX_PHRASE_LENGTH = len_comp[0]
        self.MAX_QUESTION_LENGTH = len_comp[1]
        self.MAX_ANSWER_LENGTH = 1

    def model(self):
        '''
        This function is constructing the recurrent neural network for language modeling.
        Here basic idea is phrases comprise a story. Then question is a query on that story.
        so basically we are trying to find the P(Answer|Story.Question)
        Story -> MAX_NUM_PHRASES Similar to Question
        Size of Story -> MAX_NUM_PHRASES*MAX_PHRASE_LENGTH i.e. we are picking maximum n phrases which are similar to question
        One LSTM is of phrases
        One LSTM is of question
        Merged LSTM is of questions & storie
        :return:
        '''
        phrases, questions, answers = PreprocessingUtilities.get_index_encoding_of_data(self.list_image_data, self.word_idx, self.MAX_NUM_OF_PHARSES, self.MAX_PHRASE_LENGTH, self.MAX_QUESTION_LENGTH, self.MAX_ANSWER_LENGTH)
        phrases_training = phrases[:len(phrases)/2]
        questions_training = questions[:len(questions)/2]
        answers_training = answers[:len(answers)/2]

        phrases_test = phrases[len(phrases)/2:]
        questions_test = questions[len(questions)/2:]
        answers_test = answers[len(answers)/2:]

        phrasernn = Sequential()
        phrasernn.add(Embedding(len(self.vocabulory)+1, self.EMBED_HIDDEN_SIZE,
                              input_length=self.MAX_PHRASE_LENGTH*self.MAX_NUM_OF_PHARSES))
        phrasernn.add(Dropout(0.3))


        qrnn = Sequential()
        qrnn.add(Embedding(len(self.vocabulory)+1, self.EMBED_HIDDEN_SIZE,
                           input_length=self.MAX_QUESTION_LENGTH))
        qrnn.add(Dropout(0.3))
        qrnn.add(self.RNN(self.EMBED_HIDDEN_SIZE, return_sequences=False))
        qrnn.add(RepeatVector(self.MAX_PHRASE_LENGTH*self.MAX_NUM_OF_PHARSES))

        model = Sequential()
        model.add(Merge([phrasernn, qrnn], mode='sum'))
        model.add(self.RNN(self.EMBED_HIDDEN_SIZE, return_sequences=False))
        model.add(Dropout(0.3))
        model.add(Dense(len(self.vocabulory)+1, activation='softmax'))

        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        print('Training')
        model.fit([phrases_training, questions_training], answers_training, batch_size=self.BATCH_SIZE, nb_epoch=self.EPOCHS, validation_split=0.05)
        loss, acc = model.evaluate([phrases_test, questions_test], answers_test, batch_size=self.BATCH_SIZE)
        print('Test loss / test accuracy = {:.4f} / {:.4f}'.format(loss, acc))




class MaxPoolingLangaugeModel(object):
    '''
        This class is used to form the MaxPooling neural networks and build the language models
    '''

    def __init__(self, data_set_size=100, embed_hidden_size=62, rnn=recurrent.LSTM, batch_size=32, epochs=128,
                 phrases_hidden_size=256, question_hiden_size=128, max_phrases_length=15, max_question_length=10,
                 max_answer_length=1, max_number_of_phrases=5, max_features=20000):
        self.DATASET_SIZE = data_set_size
        self.EMBED_HIDDEN_SIZE = embed_hidden_size
        self.RNN = rnn
        self.BATCH_SIZE = batch_size
        self.EPOCHS = epochs
        self.PHRASES_HIDDEN_SIZE = phrases_hidden_size
        self.QUESTION_HIDDEN_SIZE = question_hiden_size
        self.MAX_PHRASE_LENGTH = max_phrases_length
        self.MAX_QUESTION_LENGTH = max_question_length
        self.MAX_ANSWER_LENGTH = max_answer_length
        self.MAX_NUM_OF_PHARSES = max_number_of_phrases
        self.MAX_FEATURES = max_features

        self.word_idx = None
        self.list_image_data = None
        self.vocabulory = None

        # Convolution
        self.filter_length = self.MAX_PHRASE_LENGTH
        self.nb_filter = self.EMBED_HIDDEN_SIZE
        self.pool_length = self.MAX_PHRASE_LENGTH

    def preprocess_the_data(self, type_of_file=-1):
        '''
        This function will read the data from the data sets and then does the preprocessing on it so that data
        is ready to be fed to the neural network
        :param: type_of_file - This will tell which file needs to be read for the data
                By default -1: any user given file
                1 - Only Why questions
                2 - Only What questions
                3 - Only Where questions
                4 - Only Which questions
                5 - Only When questions
                6 - Only Who questions
                7 - Only How questions
                8 or other - random some questions like 100, 1000, 5000, 10000, 50000, all
        :return: -
        '''
        self.list_image_data = None
        if type_of_file == -1:
            # uncomment the line based on the size of data you want to use for training.
            # self.list_image_data = DataFileInputOutput.read_from_sample_file('sample_100.json')    #Data size 100
            # self.list_image_data = DataFileInputOutput.read_from_sample_file('sample_1000.json')   #Data size 1000
            # self.list_image_data = DataFileInputOutput.read_from_sample_file('sample_5000.json')   #Data size 5000
            # self.list_image_data = DataFileInputOutput.read_from_sample_file('sample_10000.json')  #Data size 10000
            # self.list_image_data = DataFileInputOutput.read_from_sample_file('sample_50000.json')  #Data size 50000
            # self.list_image_data = DataFileInputOutput.read_from_sample_file('sample_all.json')     #Data size everything
            # self.list_image_data = DataFileInputOutput.read_from_sample_file('sample_why.json')                           #Only why questions
            # self.list_image_data = DataFileInputOutput.read_from_sample_file('sample_what.json', elements_count=50000)    #Only what questions
            # self.list_image_data = DataFileInputOutput.read_from_sample_file('sample_where.json', elements_count=50000)   #Only where questions
            # self.list_image_data = DataFileInputOutput.read_from_sample_file('sample_which.json', elements_count=50000)   #Only which questions
            # self.list_image_data = DataFileInputOutput.read_from_sample_file('sample_when.json', elements_count=50000)    #Only when questions
            self.list_image_data = DataFileInputOutput.read_from_sample_file('sample_who.json', elements_count=50000)       #Only who questions
            # self.list_image_data = DataFileInputOutput.read_from_sample_file('sample_how.json', elements_count=50000)     #Only how questions
        else:
            if type_of_file == 1:
                print '=================================WHY QUESTION=================================================='
                self.list_image_data = DataFileInputOutput.read_from_sample_file('sample_why.json')
            elif type_of_file == 2:
                print '=================================WHAT QUESTION=================================================='
                self.list_image_data = DataFileInputOutput.read_from_sample_file('sample_what.json',
                                                                                 elements_count=50000)
            elif type_of_file == 3:
                print '=================================WHERE QUESTION=================================================='
                self.list_image_data = DataFileInputOutput.read_from_sample_file('sample_where.json',
                                                                                 elements_count=50000)
            elif type_of_file == 4:
                print '=================================WHICH QUESTION=================================================='
                self.list_image_data = DataFileInputOutput.read_from_sample_file('sample_which.json',
                                                                                 elements_count=50000)
            elif type_of_file == 5:
                print '=================================WHEN QUESTION=================================================='
                self.list_image_data = DataFileInputOutput.read_from_sample_file('sample_when.json',
                                                                                 elements_count=50000)
            elif type_of_file == 6:
                print '=================================WHO QUESTION=================================================='
                self.list_image_data = DataFileInputOutput.read_from_sample_file('sample_who.json',
                                                                                 elements_count=50000)
            elif type_of_file == 7:
                print '=================================HOW QUESTION=================================================='
                self.list_image_data = DataFileInputOutput.read_from_sample_file('sample_how.json',
                                                                                 elements_count=50000)
            else:
                print '=================================SAMPLE QUESTION=================================================='
                # self.list_image_data = DataFileInputOutput.read_from_sample_file('sample_100.json')    #Data size 100
                # self.list_image_data = DataFileInputOutput.read_from_sample_file('sample_1000.json')   #Data size 1000
                # self.list_image_data = DataFileInputOutput.read_from_sample_file('sample_5000.json')   #Data size 5000
                # self.list_image_data = DataFileInputOutput.read_from_sample_file('sample_10000.json')  #Data size 10000
                self.list_image_data = DataFileInputOutput.read_from_sample_file('sample_50000.json')  #Data size 50000
                # self.list_image_data = DataFileInputOutput.read_from_sample_file('sample_all.json')     #Data size everything

        # Get the vocabulory of the data set.
        self.vocabulory = PreprocessingUtilities.get_vocabulary_from_image_data_list(self.list_image_data)
        #Get the word to index used for encoding
        self.word_idx = PreprocessingUtilities.get_word_to_index(self.vocabulory)
        #Get the maximum length of the phrases, questions and answers
        len_comp = PreprocessingUtilities.get_max_length_components(self.list_image_data)
        self.MAX_PHRASE_LENGTH = len_comp[0]
        self.MAX_QUESTION_LENGTH = len_comp[1]
        self.MAX_ANSWER_LENGTH = 1

    def model(self):
        '''
        This function constructs the max pooling architecutre using the convolution neural networks.
        :return:
        '''
        phrases, questions, answers = PreprocessingUtilities.get_index_encoding_of_data_for_convolution(self.list_image_data,
                                                                                        self.word_idx,
                                                                                        self.MAX_NUM_OF_PHARSES,
                                                                                        self.MAX_PHRASE_LENGTH,
                                                                                       self.MAX_QUESTION_LENGTH,
                                                                                        self.MAX_ANSWER_LENGTH)
        phrases_training = phrases[:len(phrases) / 2]
        questions_training = questions[:len(questions) / 2]
        answers_training = answers[:len(answers) / 2]

        phrases_test = phrases[len(phrases) / 2:]
        questions_test = questions[len(questions) / 2:]
        answers_test = answers[len(answers) / 2:]

        phrasecnn = Sequential()
        phrasecnn.add(Embedding(len(self.vocabulory)+1, self.EMBED_HIDDEN_SIZE, input_length=self.MAX_PHRASE_LENGTH*self.MAX_NUM_OF_PHARSES))
        phrasecnn.add(Dropout(0.25))
        phrasecnn.add(Convolution1D(nb_filter=self.nb_filter,
                                filter_length=self.filter_length,
                                border_mode='valid',
                                activation='relu',
                                subsample_length=1))
        phrasecnn.add(MaxPooling1D(pool_length=self.pool_length))
        #phrasecnn.add(self.RNN(self.MAX_PHRASE_LENGTH))
        phrasecnn.add(Flatten())
        phrasecnn.add(Dense(self.EMBED_HIDDEN_SIZE))
        #phrasecnn.add(RepeatVector(self.MAX_PHRASE_LENGTH))

        qrnn = Sequential()
        qrnn.add(Embedding(len(self.vocabulory) + 1, self.EMBED_HIDDEN_SIZE,
                           input_length=self.MAX_QUESTION_LENGTH))
        qrnn.add(Dropout(0.3))
        qrnn.add(self.RNN(self.EMBED_HIDDEN_SIZE, return_sequences=False))
        qrnn.add(RepeatVector(self.MAX_PHRASE_LENGTH))

        model = Sequential()
        model.add(Merge([phrasecnn, qrnn], mode='sum'))
        model.add(self.RNN(self.EMBED_HIDDEN_SIZE, return_sequences=False))
        model.add(Dropout(0.3))
        model.add(Dense(len(self.vocabulory) + 1, activation='softmax'))

        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        print('Training')
        model.fit([phrases_training, questions_training], answers_training, batch_size=self.BATCH_SIZE,
                  nb_epoch=self.EPOCHS, validation_split=0.05)
        loss, acc = model.evaluate([phrases_test, questions_test], answers_test, batch_size=self.BATCH_SIZE)
        print('Test loss / test accuracy = {:.4f} / {:.4f}'.format(loss, acc))



