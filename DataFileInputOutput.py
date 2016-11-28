#######################################################################################################################
# Name: DataFileInputOutput
# Description: This file has classes and functions which will help you to read the data from the data file or from
#               corpora. It will also help you to generate sample files from the corpora.
# references: -
#######################################################################################################################

from ImageDataStructure import ImageData
import json


class DataFileInputOutput(object):
    '''
    This class has function which is used to intereact with the data file.
    Like reading the data file and forming the list of ImageData or using the
    given list of ImageData write json file
    '''
    data_folder = 'data\\'
    question_answer_file = 'question_answers.json'
    description_file = 'region_descriptions.json'
    @staticmethod
    def read_data_from_file(num_of_data_elements=-1, type_of_question=None):
        '''
        This function will read the actual GNOME data
        Once read it will form the list of image data - List of phrases, question and answer
        :param num_of_data_elements: Number of ImageData elements to be returned
                                     after reading the files
               typeofquestion: Type of question to be used in the data set -> where, how, why, what, who, which
        :return: list of ImageData elements
        '''
        list_image_data = []
        qa_data = None
        region_data = None
        count = 0

        # read the question answer data file, load the json data into variable
        with open(DataFileInputOutput.data_folder+DataFileInputOutput.question_answer_file) as qa_file:
            qa_data = json.load(qa_file)

        # read the phrases or image description data file, load the json data into variable
        with open(DataFileInputOutput.data_folder+DataFileInputOutput.description_file) as region_file:
            region_data = json.load(region_file)

        for i in range(0, len(region_data)):
            listQ = []
            listA = []
            listP = []
            qas = qa_data[i]['qas']
            phr = ''
            regions = region_data[i]['regions']
            if len(qas) == 0:
                continue
            for qas_dict in qas:
                listQ.append(qas_dict['question'].strip())
                listA.append(qas_dict['answer'].strip(" ."))
            for reg in regions:
                phr = reg['phrase'].strip()
                #phrshortened = removeStopWords(phr)
                if not listP.__contains__(phr):
                    listP.append(phr)
            for i in xrange(len(listQ)):
                if type_of_question is not None:
                    if not listQ[i].lower().startswith(type_of_question):
                        continue
                new_image_data = ImageData(listP, listQ[i], listA[i])
                list_image_data.append(new_image_data)
                count += 1
                if num_of_data_elements != -1 and count == num_of_data_elements:
                    break
            if num_of_data_elements != -1 and count == num_of_data_elements:
                break

        return list_image_data

    @staticmethod
    def form_sample_data_files(file_name='sample_size.json', data_elements=None, element_size=-1):
        '''
        This function will write a json file based on the data sent to it.
        This function can be used to form small data sample files.
        :param file_name: The file name to be created
        :param element_size:
        :return:
        '''
        if data_elements is None:
            print 'ERROR::No data elements passed to the function.'
            return

        with open(DataFileInputOutput.data_folder+file_name, 'wb') as open_file:
            list_details = []
            count = 0
            for details in data_elements:
                each_element = {}
                each_element.__setitem__('phrases', details.phrases)
                each_element.__setitem__('question', details.question)
                each_element.__setitem__('answer', details.answer)
                list_details.append(each_element)
                count += 1
                if element_size != -1 and element_size == count:
                    break
            json.dump(list_details, open_file)
        pass

    @staticmethod
    def read_from_sample_file(file_name='sample_size.json', elements_count=-1):
        '''
        This function will read the data from the given sample file. Then return a
        list of image-data of given count
        :param file_name: This is the file from where you need to read the data
                            by default we are using sample_size.json file in data folder
        :param elements_count: This is the count of image data to read from the file.
        :return: list of image_data of size given by elements_count. If elements_count is -1
                 then read complete file and return everything
        '''
        list_image_details = []
        counter = 0
        with open(DataFileInputOutput.data_folder+file_name) as sample:
            jdata = json.load(sample)

            for data in jdata:
                new_data = ImageData(data["phrases"], data["question"], data["answer"])
                list_image_details.append(new_data)
                counter += 1
                if elements_count != -1 and counter == elements_count:
                    break

        return list_image_details

    @staticmethod
    def generate_all_sample_data_files(type_of_file='all'):
        '''
        This function will read the the source data files from visual Gnome and generate these sub files
        So that we can only test with only few data sets. To speed testing and also for demo.
        :type_of_file: This will specify what type of file to generate
        :return:
        '''
        if type_of_file == 'all':
            # Generate a sample file having only 100 questions
            DataFileInputOutput.form_sample_data_files('sample_100.json', DataFileInputOutput.read_data_from_file(100))
            # Generate a sample file having only 1000 questions
            DataFileInputOutput.form_sample_data_files('sample_1000.json', DataFileInputOutput.read_data_from_file(1000))
            # Generate a sample file having only 5000 questions
            DataFileInputOutput.form_sample_data_files('sample_5000.json', DataFileInputOutput.read_data_from_file(5000))
            # Generate a sample file having only 10000 questions
            DataFileInputOutput.form_sample_data_files('sample_10000.json', DataFileInputOutput.read_data_from_file(10000))
            # Generate a sample file having only 50000 questions
            DataFileInputOutput.form_sample_data_files('sample_50000.json', DataFileInputOutput.read_data_from_file(50000))
            # Generate a sample file having all the questions present in the data set
            DataFileInputOutput.form_sample_data_files('sample_all.json', DataFileInputOutput.read_data_from_file())
            # Generate a sample file having all the where questions present in the data set
            DataFileInputOutput.form_sample_data_files('sample_where.json', DataFileInputOutput.read_data_from_file(type_of_question='where'))
            # Generate a sample file having all the why questions present in the data set
            DataFileInputOutput.form_sample_data_files('sample_why.json', DataFileInputOutput.read_data_from_file(type_of_question='why'))
            # Generate a sample file having all the who questions present in the data set
            DataFileInputOutput.form_sample_data_files('sample_who.json', DataFileInputOutput.read_data_from_file(type_of_question='who'))
            # Generate a sample file having all the how questions present in the data set
            DataFileInputOutput.form_sample_data_files('sample_how.json', DataFileInputOutput.read_data_from_file(type_of_question='how'))
            # Generate a sample file having all the which questions present in the data set
            DataFileInputOutput.form_sample_data_files('sample_which.json', DataFileInputOutput.read_data_from_file(type_of_question='which'))
            # Generate a sample file having all the what questions present in the data set
            DataFileInputOutput.form_sample_data_files('sample_what.json', DataFileInputOutput.read_data_from_file(type_of_question='what'))
            # Generate a sample file having all the when questions present in the data set
            DataFileInputOutput.form_sample_data_files('sample_when.json',
                                                       DataFileInputOutput.read_data_from_file(type_of_question='when'))
        elif type_of_file=='100':
            # Generate a sample file having only 100 questions
            DataFileInputOutput.form_sample_data_files('sample_100.json', DataFileInputOutput.read_data_from_file(100))
        elif type_of_file=='1000':
            # Generate a sample file having only 1000 questions
            DataFileInputOutput.form_sample_data_files('sample_1000.json',
                                                       DataFileInputOutput.read_data_from_file(1000))
        elif type_of_file=='5000':
            # Generate a sample file having only 5000 questions
            DataFileInputOutput.form_sample_data_files('sample_5000.json',
                                                       DataFileInputOutput.read_data_from_file(5000))
        elif type_of_file=='10000':
            # Generate a sample file having only 10000 questions
            DataFileInputOutput.form_sample_data_files('sample_10000.json',
                                                       DataFileInputOutput.read_data_from_file(10000))
        elif type_of_file=='50000':
            # Generate a sample file having only 50000 questions
            DataFileInputOutput.form_sample_data_files('sample_50000.json',
                                                       DataFileInputOutput.read_data_from_file(50000))
        elif type_of_file=='full_data':
            # Generate a sample file having all the questions present in the data set
            DataFileInputOutput.form_sample_data_files('sample_all.json', DataFileInputOutput.read_data_from_file())
        elif type_of_file=='why':
            # Generate a sample file having all the why questions present in the data set
            DataFileInputOutput.form_sample_data_files('sample_why.json',
                                                       DataFileInputOutput.read_data_from_file(type_of_question='why'))
        elif type_of_file=='which':
            # Generate a sample file having all the which questions present in the data set
            DataFileInputOutput.form_sample_data_files('sample_which.json', DataFileInputOutput.read_data_from_file(
                type_of_question='which'))
        elif type_of_file=='what':
            # Generate a sample file having all the what questions present in the data set
            DataFileInputOutput.form_sample_data_files('sample_what.json',
                                                       DataFileInputOutput.read_data_from_file(type_of_question='what'))
        elif type_of_file=='who':
            # Generate a sample file having all the who questions present in the data set
            DataFileInputOutput.form_sample_data_files('sample_who.json',
                                                       DataFileInputOutput.read_data_from_file(type_of_question='who'))
        elif type_of_file=='when':
            # Generate a sample file having all the when questions present in the data set
            DataFileInputOutput.form_sample_data_files('sample_when.json',
                                                       DataFileInputOutput.read_data_from_file(type_of_question='when'))
        elif type_of_file=='where':
            # Generate a sample file having all the where questions present in the data set
            DataFileInputOutput.form_sample_data_files('sample_where.json', DataFileInputOutput.read_data_from_file(
                type_of_question='where'))
        elif type_of_file=='how':
            # Generate a sample file having all the how questions present in the data set
            DataFileInputOutput.form_sample_data_files('sample_how.json',
                                                       DataFileInputOutput.read_data_from_file(type_of_question='how'))
        else:
            # your own file generation function???
            DataFileInputOutput.form_sample_data_files('sample_all.json', DataFileInputOutput.read_data_from_file())




