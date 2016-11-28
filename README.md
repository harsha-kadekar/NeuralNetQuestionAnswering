# NeuralNetQuestionAnswering
This repo has code for question answering... Given a set of phrases, based on that if a question is asked then it will give an answer


# 1st Step -> Download the data file from visual gnome website (https://visualgenome.org/)
               1. Region Description: https://visualgenome.org/static/data/dataset/region_descriptions.json.zip
               2. Question Answering: https://visualgenome.org/static/data/dataset/question_answers.json.zip

# 2nd Step -> Extract the region_descriptions.json.zip and question_answers.json.zip. 
              You will get region_descriptions.json and question_answers.json files. Place those json files inside data folder
               data folder should be present in the same folder as these scripts are present.

# 3rd Step -> Make sure you have all the necessary scripts. If not download from the git
               1. ImageDataStructure.py
               2. DataFileInputOuput.py
               3. Models.py
               4. preprocessdata.py
               5. TestScript.py

# 4th Step -> Make sure all the dependent python libraries and packages are installed
               1. Keras
               2. Theano
               3. TensorFlow
               4. NLTK
               5. Numpy
               6. itertools
               7. sklearn

# 5th Step -> Generating sample json files.
              Using this code, we can generate all the different types of sample file
                   DataFileInputOutput.generate_all_sample_data_files(type_of_file='all')
               If you want to generate specific type of file then pass correct parameter to function
                   'all' -> all the sample files will be generated
                   '100' -> sample file will have 100 questions
                   '1000' -> sample file will have 1000 questions
                   '5000' -> sample file will have 5000 questions
                   '10000' -> sample file will have 10000 questions
                   '50000' -> sample file will have 50000 questions
                   'who' -> sample file will have all the who questions
                   'what' -> sample file will have all the what questions
                   'where' -> sample file will have all the where questions
                   'which' -> sample file will have all the which questions
                   'why' -> sample file will have all the why questions
                   'how' -> sample file will have all the how questions
                   'full_data' -> sample file will have all the questions

# 6th Step -> Read data, preprocess the data, build model, train it and test the model
               First thing is you need to decide which model you are trying to build or test
               We have two models -
                   a. Story Query Model                -> StoryQueryLanguageModel
                   b. Max pooling & Convolution Model  -> MaxPoolingLangaugeModel
               As of now Story Query Model is giving good result. Max pooling model is still under construction.
               If you have selected Story Query Model following are the steps
##                   a. Instantiation of the model class
                       storymodel = StoryQueryLanguageModel()
##                   b. Reading of the file and preprocessing the data
                       storymodel.preprocess_the_data()
                       Here by default it will read sample_5000.json that is sample file containing 5000 questions.
                       You can specify what type of data file to be used for reading
                       1 - sample_why.json
                       2 - sample_what.json
                       3 - sample_where.json
                       4 - sample_which.json
                       5 - sample_when.json
                       6 - sample_who.json
                       7 - sample_how.json
                       8 - sample_100.json or sample_1000.json or sample_5000.json or sample_10000.json
                               or sample_50000.json or sample_all.json

                       note - Here for type of questions like why what where who when which and how we are
                                 only reading 50000 questions max. This is done to avoid out of memory exception.
                               If you have super computer then go ahead remove that restriction in the Models.py script
##                   c. Construct model, train the model and finally evaluate the model
                       After reading data and preprocessed it, now its time to construct the model
                           storymodel.model()
                       This function will construct the model, train it and then evaluate it based on read data.

