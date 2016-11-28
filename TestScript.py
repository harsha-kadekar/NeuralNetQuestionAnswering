#######################################################################################################################
# Name: TestScript.py
# Description: This file can be used for generating the sample data files and also to run the neural network. Train
#               and evaluate the models.
# Reference: -
#######################################################################################################################

from DataFileInputOutput import DataFileInputOutput
from Models import StoryQueryLanguageModel, MaxPoolingLangaugeModel


######################################################################################################################
# 1st Step -> Download the data file from visual gnome website (https://visualgenome.org/)
#               1. Region Description: https://visualgenome.org/static/data/dataset/region_descriptions.json.zip
#               2. Question Answering: https://visualgenome.org/static/data/dataset/question_answers.json.zip

# 2nd Step -> Extract the region_descriptions.json.zip and question_answers.json.zip. You will get
#               region_descriptions.json and question_answers.json files. Please those json files inside data folder
#               data folder should be present in the same folder as these scripts are present.

# 3rd Step -> Make sure you have all the necessary scripts. If not download from the git
#               1. ImageDataStructure.py
#               2. DataFileInputOuput.py
#               3. Models.py
#               4. preprocessdata.py
#               5. TestScript.py

# 4th Step -> Make sure all the dependent python libraries and packages are installed
#               1. Keras
#               2. Theano
#               3. TensorFlow
#               4. NLTK
#               5. Numpy
#               6. itertools
#               7. sklearn

# 5th Step -> Generating sample json files.
#               Using this code, we can generate all the different types of sample file
#                   DataFileInputOutput.generate_all_sample_data_files(type_of_file='all')
#               If you want to generate specific type of file then pass correct parameter to function
#                   'all' -> all the sample files will be generated
#                   '100' -> sample file will have 100 questions
#                   '1000' -> sample file will have 1000 questions
#                   '5000' -> sample file will have 5000 questions
#                   '10000' -> sample file will have 10000 questions
#                   '50000' -> sample file will have 50000 questions
#                   'who' -> sample file will have all the who questions
#                   'what' -> sample file will have all the what questions
#                   'where' -> sample file will have all the where questions
#                   'which' -> sample file will have all the which questions
#                   'why' -> sample file will have all the why questions
#                   'how' -> sample file will have all the how questions
#                   'full_data' -> sample file will have all the questions

# 6th Step -> Read data, preprocess the data, build model, train it and test the model
#               First thing is you need to decide which model you are trying to build or test
#               We have two models -
#                   a. Story Query Model                -> StoryQueryLanguageModel
#                   b. Max pooling & Convolution Model  -> MaxPoolingLangaugeModel
#               As of now Story Query Model is giving good result. Max pooling model is still under construction.
#               If you have selected Story Query Model following are the steps
#                   a. Instantiation of the model class
#                       storymodel = StoryQueryLanguageModel()
#                   b. Reading of the file and preprocessing the data
#                       storymodel.preprocess_the_data()
#                       Here by default it will read sample_5000.json that is sample file containing 5000 questions.
#                       You can specify what type of data file to be used for reading
#                       1 - sample_why.json
#                       2 - sample_what.json
#                       3 - sample_where.json
#                       4 - sample_which.json
#                       5 - sample_when.json
#                       6 - sample_who.json
#                       7 - sample_how.json
#                       8 - sample_100.json or sample_1000.json or sample_5000.json or sample_10000.json
#                               or sample_50000.json or sample_all.json
#
#                       note - Here for type of questions like why what where who when which and how we are
#                                 only reading 50000 questions max. This is done to avoid out of memory exception.
#                               If you have super computer then go ahead remove that restriction in the Models.py script
#                   c. Construct model, train the model and finally evaluate the model
#                       After reading data and preprocessed it, now its time to construct the model
#                           storymodel.model()
#                       This function will construct the model, train it and then evaluate it based on read data.
######################################################################################################################


def story_model_testing():
    # 6.a.1 Step - StoryQuery model class object creation
    storymodel = StoryQueryLanguageModel()

    # 6.b.1. Step - StoryQuery preprocessing of 5000 questions
    # storymodel.preprocess_the_data()
    # 6.b.1. Step - StoryQuery preprocessing of why questions
    # storymodel.preprocess_the_data(1)
    # 6.b.1. Step - StoryQuery preprocessing of what questions
    # storymodel.preprocess_the_data(2)
    # 6.b.1. Step - StoryQuery preprocessing of where questions
    # storymodel.preprocess_the_data(3)
    # 6.b.1. Step - StoryQuery preprocessing of which questions
    # storymodel.preprocess_the_data(4)
    # 6.b.1. Step - StoryQuery preprocessing of when questions
    storymodel.preprocess_the_data(5)
    # 6.b.1. Step - StoryQuery preprocessing of who questions
    # storymodel.preprocess_the_data(6)
    # 6.b.1. Step - StoryQuery preprocessing of how questions
    # storymodel.preprocess_the_data(7)
    # 6.b.1. Step - StoryQuery preprocessing of all questions
    # storymodel.preprocess_the_data(8)

    # 6.c.1. Step - StoryQuery model creation training and evaluation
    storymodel.model()


def pooling_model_testing():
    # 6.a.2 Step - pooling model class object creation
    poolmodel = MaxPoolingLangaugeModel()

    # 6.b.2. Step - pooling model preprocessing of 5000 questions
    # poolmodel.preprocess_the_data()
    # 6.b.2. Step - pooling model preprocessing of why questions
    # poolmodel.preprocess_the_data(1)
    # 6.b.2. Step - pooling model preprocessing of what questions
    # poolmodel.preprocess_the_data(2)
    # 6.b.2. Step - pooling model preprocessing of where questions
    # poolmodel.preprocess_the_data(3)
    # 6.b.2. Step - pooling model preprocessing of which questions
    # poolmodel.preprocess_the_data(4)
    # 6.b.2. Step - pooling model preprocessing of when questions
    poolmodel.preprocess_the_data(5)
    # 6.b.2. Step - pooling model preprocessing of who questions
    # poolmodel.preprocess_the_data(6)
    # 6.b.2. Step - pooling model preprocessing of how questions
    # poolmodel.preprocess_the_data(7)
    # 6.b.2. Step - pooling model preprocessing of all questions
    # poolmodel.preprocess_the_data(8)

    # 6.c.2 Step - Pooling model creation training and evalution
    poolmodel.model()


def main_function():
    # 5th Step - generating the data files consumed by out nets
    DataFileInputOutput.generate_all_sample_data_files(type_of_file='all')

    # This function will test you story query model
    story_model_testing()

    # This function will test your pooling model
    pooling_model_testing()

    pass

if __name__ == '__main__':
    main_function()