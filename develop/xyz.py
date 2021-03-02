import textract as tx
import pandas as pd
import numpy as np
import re
import os

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from polyfuzz import PolyFuzz

directory = '../data/23114.pdf'

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
    return lst

# data_dict = dict_titles_with_values(data)
# vectorizer = CountVectorizer(max_features=1500, min_df=5, max_df=0.7, stop_words=stopwords.words('english'))
# print(data_dict)

sds_official = {
    'identification of the substance/mixture and of the company/undertaking': ['product identifier','relevant identified uses of the substance or mixture and uses advised against','details of the supplier of the safety data sheet','emergency telephone number'],
    'hazards identification': ['classification of the substance or mixture', 'label elements','other hazards'],
    'composition/information on ingredients':['substances','mixtures'],
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

def tokenize_words_sds(dict):
    '''
    - Convert dictionary to dataframe
    - tokenize the sentences
    - remove stop words
    - Lemmatize words
    '''

    dict = pd.DataFrame.from_dict(dict,orient='index').T
    dict = dict.fillna(value='')
    n_columns = len(dict.columns)

    stop_words = stopwords.words('english')
    sw_list = ["the", "and", "or", "of", "?", ","]
    stop_words.extend(sw_list)

    def lemmatize_text(text):
        lemmatizer = WordNetLemmatizer()
        return [lemmatizer.lemmatize(w) for w in word_tokenize(text)]

    for i in dict:
        dict[f'{i}_No_SW'] = dict[f'{i}'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))
    dict = dict.iloc[:,n_columns:]

    for i in dict:
        dict[f'{i}_token_lemm'] = dict[i].apply(lemmatize_text)
    dict = dict.iloc[:,n_columns:]

    '''
    Renaming columns by titles that don't have stopwords in them, and that are lemmatized.
    '''
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

pdf_columns = list(data.columns)
sds_sheet_columns = list(sds_tokenized.columns)

model = PolyFuzz("TF-IDF")
model.match(pdf_columns, sds_sheet_columns)

print(model.get_matches())












































# tokenizer = Tokenizer()
# tokenizer.fit_on_texts(sds_tokenized)
# tokenizer_documents = tokenizer.texts_to_sequences(sds_tokenized)
# tokenized_paded_documents = pad_sequences(tokenizer_documents, maxlen=64, padding='post')
# vocab_size = len(tokenizer.word_index)+1
# # print(tokenizer_paded_documents)
#
# W2V_PATH="../GoogleNews-vectors-negative300.bin.gz"
# model_w2v = gensim.models.KeyedVectors.load_word2vec_format(W2V_PATH, binary=True)
#
# # creating embedding matrix, every row is a vector representation from the vocabulary indexed by the tokenizer index.
# embedding_matrix=np.zeros((vocab_size,300))
# for word,i in tokenizer.word_index.items():
#     if word in model_w2v:
#         embedding_matrix[i]=model_w2v[word]
# # creating document-word embeddings
# document_word_embeddings=np.zeros((len(tokenized_paded_documents),64,300))
# for i in range(len(tokenized_paded_documents)):
#     for j in range(len(tokenized_paded_documents[0])):
#         document_word_embeddings[i][j]=embedding_matrix[tokenized_paded_documents[i][j]]
# # print(document_word_embeddings.shape)
#
# document_embeddings=np.zeros((len(tokenized_paded_documents),300))
# tfidvectorizer = TfidfVectorizer()
# words=tfidvectorizer.get_feature_names()
# for i in range(len(document_word_embeddings)):
#     for j in range(len(words)):
#         document_embeddings[i]+=embedding_matrix[tokenizer.word_index[words[j]]]*tfidf_vectors[i][j]
#
# print(document_embeddings.shape)
# pairwise_similarities=cosine_similarity(document_embeddings)
# pairwise_differences=euclidean_distances(document_embeddings)
#
# print(most_similar(0,pairwise_similarities,'Cosine Similarity'))
# print(most_similar(0,pairwise_differences,'Euclidean Distance'))