#######################################################################################################################
# Name: ImageDataStructure.py
# Description: This file has functions and classes which will define our data format. So when we read the data from
#               corpora or dataset will be converting to the format defined in this file.
# references: -
#######################################################################################################################

class ImageData(object):
    '''
    This class is used to represent for data representation.
    Every data point will have an object of this class
    Every data point will have a following things
        1. List of phrases
        2. A question
        3. An answer
    '''
    def __init__(self, phrases=[], quest=None, ans=None):
        self._phrases = phrases
        self._question = quest
        self._answer = ans

    @property
    def phrases(self):
        return self._phrases

    @property
    def question(self):
        return self._question

    @property
    def answer(self):
        return self._answer
