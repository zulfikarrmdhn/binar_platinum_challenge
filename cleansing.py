from unidecode import unidecode
import re
import pandas as pd

def lowercase(s):
    return s.lower()

def replace_ascii(s):
    return unidecode(s)

def remove_ascii2(s):
    return re.sub(r'\\x[A-Za-z0-9./]+', ' ', unidecode(s))

def remove_n(s):
    return re.sub(r'\n', ' ', s)

def remove_punct(s):
    return re.sub(r'[^\w\d\s]+', ' ',s)

def remove_whitespace(s):
    return re.sub(r' +', ' ', s)

alay_dict = pd.read_csv('Data/new_kamusalay.csv', encoding='latin-1', header=None)
alay_dict = alay_dict.rename(columns={0:'original',1:'replacement'})
alay_dict_map = dict(zip(alay_dict['original'], alay_dict['replacement']))

def alay_word(s):
    return ' '.join([alay_dict_map[word] if word in alay_dict_map else word for word in s.split(' ')])

abusive = pd.read_csv('Data/abusive.csv', encoding='latin-1')
abusive_map = abusive['ABUSIVE'].str.lower().tolist()

def abusive_word (s):
    word_list = s.split()
    return ' '.join([s for s in word_list if s not in abusive_map ])

def cleansing(s):
    s = remove_ascii2(s)
    s = remove_n(s)
    s = remove_punct(s)
    s = remove_whitespace(s)
    s = alay_word(s)
    s = abusive_word(s)
    return s