#######################################################################################################################
# Name: preprocessdata.py
# Description: This file has functions and classes which can be used for preprocess the data like converting them into
#               vectors, tokenizing the sentences.
# References:
#######################################################################################################################
from ImageDataStructure import ImageData
from DataFileInputOutput import DataFileInputOutput
from keras.preprocessing.text import Tokenizer
from keras.utils.np_utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.layers.embeddings import Embedding
import nltk
import numpy as np
import itertools
import string
from sklearn.feature_extraction.text import TfidfVectorizer

class PreprocessingUtilities(object):
    '''
    This class will have functions which will help in preprocessing the given strings.
    It can also transform from one form of input to other.
    '''
    #TODO:Can we use already existing stopwords present in standard libraries like NLTK??
    stopwords = ['a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 'any', 'are', 'aren\'t',
                 'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by',
                 'can\'t', 'cannot', 'could', 'couldn\'t', 'did', 'didn\'t', 'do', 'does', 'doesn\'t', 'doing',
                 'don\'t', 'down', 'during', 'each', 'few', 'for', 'from', 'further', 'had', 'hadn\'t', 'has',
                 'hasn\'t', 'have', 'haven\'t', 'having', 'he', 'he\'d', 'he\'ll', 'he\'s', 'her', 'here', 'here\'s',
                 'hers', 'herself', 'him', 'himself', 'his', 'how', 'how\'s', 'i', 'i\'d', 'i\'ll', 'i\'m', 'i\'ve',
                 'if', 'in', 'into', 'is', 'isn\'t', 'it', 'it\'s', 'its', 'itself', 'let\'s', 'me', 'more', 'most',
                 'mustn\'t', 'my', 'myself', 'no', 'nor', 'not', 'of', 'off', 'on', 'once', 'only', 'or', 'other',
                 'ought', 'our', 'ours ourselves', 'out', 'over', 'own', 'same', 'shan\'t', 'she', 'she\'d', 'she\'ll',
                 'she\'s', 'should', 'shouldn\'t', 'so', 'some', 'such', 'than', 'that', 'that\'s', 'the', 'their',
                 'theirs', 'them', 'themselves', 'then', 'there', 'there\'s', 'these', 'they', 'they\'d', 'they\'ll',
                 'they\'re', 'they\'ve', 'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up', 'very', 'was',
                 'wasn\'t', 'we', 'we\'d', 'we\'ll', 'we\'re', 'we\'ve', 'were', 'weren\'t', 'what', 'what\'s', 'when',
                 'when\'s', 'where', 'where\'s', 'which', 'while', 'who', 'who\'s', 'whom', 'why', 'why\'s', 'with',
                 'won\'t', 'would', 'wouldn\'t', 'you', 'you\'d', 'you\'ll', 'you\'re', 'you\'ve', 'your', 'yours',
                 'yourself', 'yourselves']
    @staticmethod
    def remove_stop_words(original_sentence):
        '''
        This function will help in removing the stop words in a given sentence.
        :param original_sentences:
        :return:
        '''
        sentence = []
        string = original_sentence.lower().split(' ')
        for s in string:
            if s not in PreprocessingUtilities.stopwords:
                sentence.append(s)

        return ' '.join(sentence)

    @staticmethod
    def get_vocabulary_from_image_data_list(list_image_data):
        '''
        This function given a list of image_data will return all the words present in the
        phrases, question and answer
        :param list_image_data: list of image data whose words needs to be listed
        :return: set of words present all of the image data.
        '''
        words = []
        previous_pharses = None
        for detail in list_image_data:
            if previous_pharses != detail.phrases:
                for phr in detail.phrases:
                    words.append(PreprocessingUtilities.tokenize_sentence(phr))
                    previous_pharses = detail.phrases
            words.append(PreprocessingUtilities.tokenize_sentence(detail.question))
            words.append(PreprocessingUtilities.tokenize_sentence(detail.answer))

        return sorted(set(itertools.chain(*words)))

    @staticmethod
    def tokenize_sentence(sentence):
        '''
        Given a sentence, this function will break that sentence into words and return the words
        of that sentence.
        :param sentance: Sentence which needs to be tokenized
        :return: list of words of that sentence
        '''
        lemma = lambda x: x.strip().lower().split(' ')
        #TODO: advanced lemmatization or tokenization can be done here. Like weedout any stopwords
        #punkt = nltk.data.load('tokenizers/punkt/english.pickle')
        #tokens_of_sentence = punkt.tokenize(sentence.lower())
        words2 = nltk.word_tokenize(sentence)
        #return lemma(sentence)
        words2 = [word for word in words2 if not all(char in string.punctuation for char in word)]
        # Remove contractions - wods that begin with '
        tokens_of_sentence = [word for word in words2 if not (word.startswith("'"))]
        return tokens_of_sentence

    @staticmethod
    def get_word_to_index(vocabulory):
        '''
        This function given a vocabulory, will return a word to index list.
        :param vocabulory: All the words in the data corpora or vocabulory
        :return: dictionary containing the word:index format
        '''
        word_idx = dict((c, i + 1) for i, c in enumerate(vocabulory))
        return word_idx

    @staticmethod
    def get_max_length_of_sentence_in_list(sentences):
        '''
        This function will return the max length among all the sentences given in the format of list
        :param sentences: list of sentences whose max length needs to be found out
        :return: length of sentences whose length is highest
        '''
        return max(map(len, (PreprocessingUtilities.tokenize_sentence(x) for x in sentences)))

    @staticmethod
    def get_max_phrase_length(list_image_data):
        '''
        This function will give the maximum length of all the phrases present in the list of image data
        :param list_image_data: list of image data among which phrases max length needs to be found out
        :return: maximum length of all the phrases
        '''
        phrases_list = []
        old_phrases = None
        for data in list_image_data:
            if old_phrases != data.phrases:
                phrases_list.extend(data.phrases)
                old_phrases = data.phrases
        return PreprocessingUtilities.get_max_length_of_sentence_in_list(phrases_list)

    @staticmethod
    def get_max_question_length(list_image_data):
        '''
        This function will give the maximum length of all the questions present in the list of image data
        :param list_image_data: list of image data among which max length of questions needs to be found out
        :return: maximum length of all the questions
        '''
        questions_list = []
        for data in list_image_data:
            questions_list.append(data.question)
        return PreprocessingUtilities.get_max_length_of_sentence_in_list(questions_list)

    @staticmethod
    def get_max_answer_length(list_image_data):
        '''
        this function will give the maximum length of all the answers present in the list of image data
        :param list_image_data: list of image data among which max length of answers needs to be found out
        :return: maximum length of all the answers
        '''
        answers_list = []
        for data in list_image_data:
            answers_list.append(data.answer)
        return PreprocessingUtilities.get_max_length_of_sentence_in_list(answers_list)

    @staticmethod
    def get_max_length_components(list_image_data):
        '''

        :param list_image_data:
        :return:
        '''
        answers_list = []
        phrases_list = []
        questions_list = []
        old_phrase = None
        for data in list_image_data:
            if old_phrase != data.phrases:
                phrases_list.extend(data.phrases)
                old_phrase = data.phrases
            questions_list.append(data.question)
            answers_list.append(data.answer)
        max_phrase_length = PreprocessingUtilities.get_max_length_of_sentence_in_list(phrases_list)
        max_question_length = PreprocessingUtilities.get_max_length_of_sentence_in_list(questions_list)
        max_answer_length = PreprocessingUtilities.get_max_length_of_sentence_in_list(answers_list)

        return (max_phrase_length, max_question_length, max_answer_length)

    @staticmethod
    def get_combined_single_phrase_index_vectors(word_idx, list_phrases, max_length_of_each_phrase):
        '''
        This function given a list of phrases and word to index, will combine them into a single phrase
        and each word is represented as index of that word to index.
        :param word_idx: dictionary of words to their index
        :param list_phrases: list of phrases which needs to be combined and encoded into a single phrases
        :param max_length_of_each_phrase: length of single phrase
        :return: list of indexes corresponding to the phrases.
        '''
        final_phrase = []
        for phrase in list_phrases:
            phrase_words = PreprocessingUtilities.tokenize_sentence(phrase)
            word_list = [word_idx[w] for w in phrase_words]
            #padded_phrase = pad_sequences(word_list, maxlen=max_length_of_each_phrase)
            if word_list.__len__() < max_length_of_each_phrase:
                while len(word_list) < max_length_of_each_phrase:
                    word_list.append(0)
            elif word_list.__len__() > max_length_of_each_phrase:
                while len(word_list) > max_length_of_each_phrase:
                    word_list.pop()
            final_phrase.extend(word_list)
        return final_phrase

    @staticmethod
    def get_convoluted_single_phrase_index_vectors(word_idx, list_phrases):
        '''
        This function will combine all the pharases into a single phrase. This will just combine every word of phrases
        without padding for each phrases. This will be used in convolutional neural network where pooling is involved.
        :param word_idx: Dictionary having words and their corresponding ids
        :param list_phrases: list of phrases which needs to be encoded
        :return: an array of word indexed sentence combining all phrases.
        '''
        final_phrase = []
        for phrase in list_phrases:
            phrase_words = PreprocessingUtilities.tokenize_sentence(phrase)
            word_list = [word_idx[w] for w in phrase_words]
            final_phrase.extend(word_list)
        return final_phrase



    @staticmethod
    def get_question_index_vectors(word_idx, question):
        '''
        This function given a question will convert into a list of index corresponding to the words in the question
        :param word_idx: dictionary of words and their corresponding indices
        :param question: question which needs to be encoded
        :return: list of indices based on the words in question and their value in word index
        '''
        question_words = PreprocessingUtilities.tokenize_sentence(question)
        word_list = [word_idx[w] for w in question_words]
        return word_list

    @staticmethod
    def get_answer_index_vectors(word_idx, answer):
        '''
        This function will give a one hot encoding of the answer. It will form an array whose size is equal to
        length of the vocabulory. Every word in the answer will be represented by 1 in the array
        :param word_idx: dictionary of words and their index
        :param answer: answer which needs to be encoded
        :return:
        '''
        y = np.zeros(len(word_idx) + 1)  # let's not forget that index 0 is reserved
        words = PreprocessingUtilities.tokenize_sentence(answer)
        for word in words:
            y[word_idx[word]] = 1
            #break                   #TODO:only one word answers. Will fix for sentence answers later
        return y

    @staticmethod
    def get_index_encoding_of_data(list_data, word_idx, max_num_phrases, max_phrase_length, max_question_length, max_answer_length):
        '''
        Given a list of image_data objects this function will convert them into word index encoding.
        :param list_data: list of image_data objects
        :param word_idx: dictionary of words and their index as keys
        :param max_num_phrases: Total number of phrases each question can have
        :param max_phrase_length: Maximum length of the single phrase
        :param max_question_length: Maximum length of the single question
        :param max_answer_length: Maximum length of the single answer.
        :return: Encoded Pharases, Encoded Questions,Encoded Answers
        '''
        phrases = []
        questions = []
        answers = []
        for data in list_data:
            list_phrases = PreprocessingUtilities.get_related_phrases_using_cosinesimilarity(data.phrases, data.question, max_num_phrases)
            phrases.append(PreprocessingUtilities.get_combined_single_phrase_index_vectors(word_idx, list_phrases, max_phrase_length))
            questions.append(PreprocessingUtilities.get_question_index_vectors(word_idx, data.question))
            answers.append(PreprocessingUtilities.get_answer_index_vectors(word_idx, data.answer))
        return pad_sequences(phrases, maxlen=max_num_phrases*max_phrase_length), pad_sequences(questions, maxlen=max_question_length), np.array(answers)#pad_sequences(answers, maxlen=max_answer_length)

    @staticmethod
    def get_index_encoding_of_data_for_convolution(list_data, word_idx, max_num_phrases, max_phrase_length, max_question_length, max_answer_length):
        '''
        Given a list of image_data objects this function will convert them into word index encoding.
        :param list_data: list of image_data objects
        :param word_idx: dictionary of words and their index as keys
        :param max_num_phrases: Total number of phrases each question can have
        :param max_phrase_length: Maximum length of the single phrase
        :param max_question_length: Maximum length of the single question
        :param max_answer_length: Maximum length of the single answer.
        :return: Encoded Pharases, Encoded Questions,Encoded Answers
        '''

        phrases = []
        questions = []
        answers = []
        for data in list_data:
            list_phrases = PreprocessingUtilities.get_related_phrases_using_cosinesimilarity(data.phrases,
                                                                                             data.question,
                                                                                             max_num_phrases)
            #phrases.append(PreprocessingUtilities.get_convoluted_single_phrase_index_vectors(word_idx, list_phrases))
            phrases.append(PreprocessingUtilities.get_combined_single_phrase_index_vectors(word_idx, list_phrases,
                                                                                           max_phrase_length))
            questions.append(PreprocessingUtilities.get_question_index_vectors(word_idx, data.question))
            answers.append(PreprocessingUtilities.get_answer_index_vectors(word_idx, data.answer))
        return pad_sequences(phrases, maxlen=max_num_phrases * max_phrase_length), pad_sequences(questions, maxlen=max_question_length), np.array(answers)  # pad_sequences(answers, maxlen=max_answer_length)

    @staticmethod
    def get_single_phrase_question_answer_format(image_data):
        pass

    @staticmethod
    def get_related_phrases_using_cosinesimilarity(list_phrases, question, max_num_phrases):
        '''
        This function will get a list of similar phrases based on the question given.
        :param list_phrases: list of phrases among which we need to choose those which are similar
        :param question: question base on which similarity needs to be decided
        :param max_num_phrases: Maximum number of similar phrases to be returned
        :return: list of phrases which are similar to question asked.
        '''
        pharses = list_phrases
        if len(list_phrases) > max_num_phrases:
            vect = TfidfVectorizer(min_df=1)
            to_compare = []
            to_compare.extend(list_phrases)
            to_compare.append(question)
            tfidf = vect.fit_transform(to_compare)
            ar = (tfidf*tfidf.T).A
            list_compare = []
            question_index = len(to_compare) - 1
            for i in xrange(len(to_compare) - 1):
                score = ar[question_index, i]
                list_compare.append((score, list_phrases[i], i))

            pharses = []

            for i in xrange(max_num_phrases):
                max_value = list_compare[0]
                index = 0
                for j in xrange(len(list_compare) - 1):
                    if list_compare[j][0] > max_value[0]:
                        max_value = list_compare[j]
                        index = j
                pharses.append(list_compare[index][1])
                del list_compare[index]

        return pharses

    @staticmethod
    def vectorize_using_word2vec(listImageDetails, max_features, max_hidden_layer, max_num_of_phrases,
                                 max_phrase_length, max_question_length, max_answer_length):
        '''
        TODO: Still work is going on in this function.
        :param listImageDetails:
        :param max_features:
        :param max_hidden_layer:
        :param max_num_of_phrases:
        :param max_phrase_length:
        :param max_question_length:
        :param max_answer_length:
        :return:
        '''
        tokenizer = Tokenizer(nb_words=max_features)
        texts = []
        list_questions = []
        list_answers = []
        list_phrases = []
        list_phrases_final = []
        old_phrase = None
        labels = []
        labels_index = {}
        count = 0
        for image_detail in listImageDetails:
            name = 'Q_' + str(count)
            label_id = len(labels_index)
            labels_index[name] = label_id
            list_phrases.append(image_detail.phrases)
            # if image_detail.phrases != old_phrase:
            texts.extend(image_detail.phrases)
            for x in image_detail.phrases:
                labels.append(label_id)
            # old_phrase = image_detail.phrases
            texts.append(image_detail.question)
            labels.append(label_id)
            list_questions.append(image_detail.question.encode('ascii'))
            texts.append(image_detail.answer)
            labels.append(label_id)
            list_answers.append(image_detail.answer.encode('ascii'))
            count += 1

        texts = [s.encode('ascii') for s in texts]
        tokenizer.fit_on_texts(texts)

        count = 0
        for phrases in list_phrases:
            rel_phrases = PreprocessingUtilities.get_related_phrases_using_cosinesimilarity(phrases,
                                                                                            list_questions[count],
                                                                                            max_num_of_phrases)
            rel_phrases = [s.encode('ascii') for s in rel_phrases]
            rel_phrases = tokenizer.texts_to_sequences(rel_phrases)
            rel_phrases = pad_sequences(rel_phrases, maxlen=max_phrase_length)
            phrase = []
            for phr in rel_phrases:
                phrase.extend(phr)
            list_phrases_final.append(phrase)
            count += 1

        list_questions = pad_sequences(tokenizer.texts_to_sequences(list_questions), maxlen=max_question_length)
        list_answers = pad_sequences(tokenizer.texts_to_sequences(list_answers), maxlen=max_answer_length)

        list_phrases_final = pad_sequences(list_phrases_final, maxlen=max_phrase_length * max_num_of_phrases)

        word_index = tokenizer.word_index
        print('Found %s unique tokens.' % len(word_index))

        labels = to_categorical(np.asarray(labels))
        # print('Shape of data tensor:', data.shape)
        # print('Shape of label tensor:', labels.shape)

        # split the data into a training set and a validation set
        # indices = np.arange(data.shape[0])
        # np.random.shuffle(indices)
        # data = data[indices]
        # labels = labels[indices]
        # nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

        # x_train = data[:-nb_validation_samples]
        # y_train = labels[:-nb_validation_samples]
        # x_val = data[-nb_validation_samples:]
        # y_val = labels[-nb_validation_samples:]

        embeddings_index = {}
        with open(DataFileInputOutput.data_folder + 'glove.6B.300d.txt') as f:
            for line in f:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs

        embedding_matrix = np.zeros((len(word_index) + 1, max_hidden_layer))
        for word, i in word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector

        phrase_embedding_layer = Embedding(len(word_index) + 1,
                                           max_hidden_layer,
                                           weights=[embedding_matrix],
                                           input_length=max_num_of_phrases * max_phrase_length,
                                           trainable=False)

        question_embedding_layer = Embedding(len(word_index) + 1,
                                             max_hidden_layer,
                                             weights=[embedding_matrix],
                                             input_length=max_question_length,
                                             trainable=False)