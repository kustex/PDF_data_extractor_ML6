#!usr/bin/python3

import textract as tx
import pandas as pd
import numpy as np
import itertools
import re
import os
import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

directory = '../data/Omnirad-184.pdf'

def check_if_title(sentence):
    '''

    Args:
        sentence: String

    Returns: A value 1 if a string is a title. Value 0 if not.

    '''
    sentence = str(sentence)
    word_list = ['identification', 'composition', 'ingredients', 'information', 'measures', 'handling', 'consideration',
                 'properties', 'stability', 'considerations', 'exposure', 'section']
    starts_with_section = False
    starts_with_number = False
    # contains_irrelevant_numbers = False
    splitted_sentence = sentence.split()
    if len(splitted_sentence) > 1:
        starts_with_section = sentence.split()[0] == 'section'
        starts_with_number = sentence[0].isdigit() and not any(char.isdigit() for char in sentence[4:])
        short_sentence = len(sentence) <= 50

    contains_title_word = any(word in splitted_sentence for word in word_list)

    if (starts_with_number or starts_with_section) and contains_title_word and short_sentence:
        return 1
    else:
        return 0

def clean_text(text):
    '''

    Args:
        text: String

    Returns: Cleaned text

    '''
    # remove unwanted chars
    text = re.sub(r'[^\w]', ' ', text)
    # remove starting and trailing spaces
    text = text.lstrip()
    text = text.rstrip()
    # make everything lowercase
    text = text.lower()
    # delete multiple spaces
    text = re.sub(' +', ' ', text)
    # limit to 256 words
    wordlist = text.split()

    words_text = len([item for item in wordlist])
    if words_text < 256:
        wordlist = [item[:] for item in wordlist]
    else:
        wordlist = [item[:256] for item in wordlist]
    text = ' '.join([str(elem) for elem in wordlist])
    return text

def df_from_text(path):
    '''

    Args:
        path: The path to a PDF-file.

    Returns: Makes dataframe out of plain text (PDF) and applies clean_text() and check_if_title() function to create a dataframe.

    '''
    text = tx.process(path)
    text = text.decode('utf-8')
    text_splitted = re.split("\r\n|\r\n\r\n", text)

    while '' in text_splitted:
        text_splitted.remove('')

    d = {'sentence': text_splitted}
    df_text = pd.DataFrame(data=d)
    df_text['title'] = 0

    df_text["sentence"] = df_text['sentence'].apply(lambda x: clean_text(x))
    df_text['title'] = df_text['sentence'].map(lambda a: check_if_title(a))
    return df_text

def dict_titles_with_values(data):
    '''

    Args:
        data: This is a Dataframe with rows that have been cleaned by 'clean_text()' and columns that have been checked by 'check_if_title()'

    Returns: a dictionary with filtered titles and corresponding corpus.

    '''
    lst = {}
    current_title = ''
    for i,j in zip(data['title'], data['sentence']):
        if i == 1:
            current_title = j
            lst[j] = []
        else:
            if current_title != '':
                lst[current_title].append(j)
    return lst

def make_pdf_dict(directory):
    pdf_dict = {}
    key_dict = {}
    for i, filename in enumerate(os.listdir(directory)):
        if filename.endswith(".pdf"):
            file = os.path.join(directory, filename)
            pdf_dict[i] = df_from_text(file)
            key_dict[i] = filename
        else:
            print("No PDF files found")
    return pdf_dict, key_dict

sds_official = {
    'identification of the substance mixture and of the company undertaking': ['company','product identifier','relevant identified uses of the substance or mixture and uses advised against','details of the supplier of the safety data sheet','emergency telephone number'],
    'hazards identification': ['classification of the substance or mixture', 'label elements','other hazards'],
    'composition information on ingredients':['substances','mixtures'],
    'first aid measures':['description of first aid measures','most important symptoms and effects, both acute and delayed','indication of any immediate medical attention and special treatment needed'],
    'firefighting measures':['extinguishing media','special hazards arising from the substance or mixture','advice for firefighters'],
    'accidental release measure':['personal precautions, protective equipment and emergency procedures','environmental precautions','methods and material for containment and cleaning up','reference to other sections'],
    'handling and storage':['precautions for safe handling','conditions for safe storage, including any incompatibilities','specific end use(s)'],
    'exposure controls/personal protection':['control parameters','exposure controls'],
    'physical and chemical properties':['information on basic physical and chemical properties','other information'],
    'stability and reactivity':['reactivity','chemical stability','possibility of hazardous reactions','conditions to avoid','incompatible materials','hazardous decomposition products'],
    'toxicological information':['information on toxicological effects'],
    'ecological information':['toxicity', 'persistence and degradability','bioaccumulative potential','mobility in soil','results of pbt and vpvb assessment','other adverse effects'],
    'disposal considerations':['waste treatment methods'],
    'transport information':['un number','un proper shipping name','transport hazard class(es)','packing group','environmental hazards','special precautions for user','transport in bulk according to annex II of marpol and the ibc code'],
    'regulatory information':['safety, health and environmental regulations/legislation specific for the substance or mixture','chemical safety assessment'],
    'other information':['date of the latest revision of the sds']
}

'''
Label all titles as keys and label values as values
then try to label the corpus in the columns as keys or values
- to label we need to vectorize the keys and values in sds_official and compare to columns.
'''

def lemmatize_text(text):
    '''

    Args:
        text:lemmatizes text

    Returns: list of words that are lemmatized

    '''
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(w) for w in word_tokenize(text)]

def tokenize_words_sds(dict):
    '''

    Args:
        dict: Dictionary

    Operation:
        - Convert dictionary to dataframe
        - tokenize the sentences
        - remove stop words
        - Lemmatize words

    Returns: Dataframe

    '''

    dict = pd.DataFrame.from_dict(dict,orient='index').T
    dict = dict.fillna(value='')
    n_columns = len(dict.columns)

    stop_words = stopwords.words('english')
    sw_list = ["the", "and", "or", "of", "?", ",", " "]
    stop_words.extend(sw_list)

    for i in dict:
        dict[f'{i}_No_SW'] = dict[f'{i}'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))
    dict = dict.iloc[:,n_columns:]

    for i in dict:
        dict[f'{i}_token_lemm'] = dict[i].apply(lemmatize_text)
    dict = dict.iloc[:,n_columns:]

    columns = [column for column in dict.columns]
    columns_no_sw_lemm = []
    for i in columns:
        tokenized_column = []
        tokenized_words = word_tokenize(i[0:-17])
        for word in tokenized_words:
            if word not in stop_words:
                tokenized_column.append(word)
        columns_no_sw_lemm.append(list(tokenized_column))

    column_names = []
    for lists_of_words in columns_no_sw_lemm:
        column_names.append(','.join(lists_of_words))
    column_names = [column_name.replace(',', " ") for column_name in column_names]
    dict.columns = column_names
    return dict

data = tokenize_words_sds(dict_titles_with_values(df_from_text(directory)))
sds_tokenized = tokenize_words_sds(sds_official)

def unique(list1):
    '''

    Args:
        list1: List of strings/values

    Returns: a list of unique values in the input list.

    '''
    # intilize a null list
    unique_list = []
    # traverse for all elements
    for x in list1:
        # check if exists in unique_list or not
        if x not in unique_list:
            unique_list.append(x)
    return unique_list

def get_list_of_columns(df_data, df_sds):
    '''

    Args:
        df_data:
        df_sds:

    Returns: It has to return a

    '''

    columns_data = [column for column in df_data.columns]
    columns_sds = [column for column in df_sds.columns]
    tokenized_columns_data = [word_tokenize(i) for i in columns_data]
    tokenized_columns_sds = [word_tokenize(i) for i in columns_sds]

    # print(tokenized_columns_data)
    # print(len(tokenized_columns_data))
    # print("--------------------")
    # print(tokenized_columns_sds)
    # print(len(tokenized_columns_sds))
    # print("--------------------")

    lst = []
    for w1 in tokenized_columns_data:
        for w2 in tokenized_columns_sds:
            if nltk.edit_distance(w1, w2) <= 2 and w1 not in lst:
                lst.append(w1)
    lst = [" ".join(word) for word in lst]

    # print(len(lst))
    # print("--------------------")
    # print(lst)
    # print("--------------------")
    # print(len(df_sds))
    return lst

# print(get_list_of_columns(data, sds_tokenized))

def get_dict_keys(df_data, df_sds):
    '''

    Args:
        df_data: This is the input pdf file as a dataframe, It has to be sorted per column that is the same column for df_sds.
        df_sds: This is the Safety Data Sheet official format file as a dataframe.

    Returns: Dictionary of df_data's filtered keys per heading.

    '''
    keyList = np.arange(0, len(df_sds))
    dicts =  dict.fromkeys(keyList, None)

    for range in keyList:
        list_of_lists = []
        for zin_a in df_sds.iloc[:,range]:
            for zin_b in df_data.iloc[:,range]:
                for word_a in zin_a:
                    for word_b in zin_b:
                        if nltk.edit_distance(word_a, word_b) <= 2 and zin_b not in list_of_lists:
                            list_of_lists.append(zin_b)

        list_of_lists_filtered = []
        for zin in list_of_lists:
            not_digits = []
            for word in zin:
                if not word.isdigit():
                    not_digits.append(word)
            list_of_lists_filtered.append(not_digits)

        list_of_lists_filtered_no_small_words = []
        for zin in list_of_lists_filtered:
            no_small = []
            for word in zin:
                if not len(word) <= 3:
                    no_small.append(word)
            list_of_lists_filtered_no_small_words.append(no_small)

        list_of_lists_no_single_words = []
        for zin in list_of_lists_filtered_no_small_words:
            no_single_word = []
            if not len(zin) <= 1:
                no_single_word.append(zin)
            list_of_lists_no_single_words.append(no_single_word)

        # lst = [" ".join(word) for word in list_of_lists_no_single_words]
        dicts[range] = list_of_lists_no_single_words
    return dicts




