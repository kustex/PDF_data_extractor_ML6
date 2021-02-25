import textract as tx
import pandas as pd
import re
import os

def check_if_title(sentence):
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

directory = '../data/23114.pdf'
data = df_from_text(directory)

def dict_titles_with_values(data):
    lst = {}
    current_title = ''
    for i,j in zip(data['title'], data['sentence']):
        if i == 1:
            current_title = j
            lst[j] = []
        else:
            if current_title != '':
                lst[current_title].append(j)
    lst = pd.DataFrame.from_dict(lst, orient='index').T
    # for column in lst:
    #     for row in column:
    #         row = lambda s: "" if s in None else str(s)
    return lst

data_dict = dict_titles_with_values(data)
print(data_dict)
