import textract as tx
import pandas as pd
import numpy as np
import itertools
import json
import re
import os
import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

labeled_headers = ['identification of the substance mixture and of the company undertaking',
                   'hazards identification',
                   'composition information on ingredients',
                   'first aid measures',
                   'firefighting measures',
                   'accidental release measure',
                   'handling and storage',
                   'exposure controls/personal protection',
                   'physical and chemical properties',
                   'stability and reactivity',
                   'toxicological information',
                   'ecological information',
                   'disposal considerations',
                   'transport information',
                   'regulatory information',
                   'other information']

labeled_keys = ['company', 'product identifier', 'relevant identified uses of the substance or mixture and uses advised against', 'details of the supplier of the safety data sheet','emergency telephone number',
                'classification of the substance or mixture', 'label elements','other hazards',
                'substances','mixtures',
                'description of first aid measures','most important symptoms and effects, both acute and delayed','indication of any immediate medical attention and special treatment needed',
                'extinguishing media','special hazards arising from the substance or mixture','advice for firefighters',
                'personal precautions, protective equipment and emergency procedures','environmental precautions','methods and material for containment and cleaning up','reference to other sections',
                'precautions for safe handling','conditions for safe storage, including any incompatibilities','specific end use(s)',
                'control parameters','exposure controls',
                'information on basic physical and chemical properties','other information',
                'reactivity','chemical stability','possibility of hazardous reactions','conditions to avoid','incompatible materials','hazardous decomposition products',
                'information on toxicological effects',
                'toxicity', 'persistence and degradability','bioaccumulative potential','mobility in soil','results of pbt and vpvb assessment','other adverse effects',
                'waste treatment methods',
                'un number','un proper shipping name','transport hazard class(es)','packing group','environmental hazards','special precautions for user','transport in bulk according to annex II of marpol and the ibc code',
                'safety, health and environmental regulations/legislation specific for the substance or mixture','chemical safety assessment',
                'date of the latest revision of the sds']

labeled_values = []

Tokenized_text = []

def lemmatize_text(text):
    '''

    Args:
        text:lemmatizes text

    Returns: list of words that are lemmatized

    '''
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(w) for w in word_tokenize(text)]

stop_words = stopwords.words('english')
sw_list = ["the", "and", "or", "of", "?", ",", " "]
stop_words.extend(sw_list)


labeled_headers_tokenized = lemmatize_text(labeled_headers)
labeled_keys_tokenized = lemmatize_text(labeled_keys)

print(labeled_headers_tokenized)























#
#
# dict = {}
#
# for word in Tokenized_text:
#     if word <= 2 levinshtein with word in labeled hearders:
#         dict.header.append(word)
#     elif word <= ... levinshtein with word in labeled keys:
#         dict.keys.append(word)
#     elif word:
#         dict.values.append(word)
