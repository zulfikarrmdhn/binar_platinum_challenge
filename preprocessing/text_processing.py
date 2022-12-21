import re
import pickle
import pandas as pd
import os
from flashtext import KeywordProcessor
from keras_preprocessing.sequence import pad_sequences

alay_dict_df = pd.read_csv('data\colloquial-indonesian-lexicon.csv')
alay_dict = alay_dict_df.groupby('formal')['slang'].apply(list).to_dict()
keyword_processor = KeywordProcessor()
keyword_processor.add_keywords_from_dict(alay_dict)

class TextProcessing():
    def __init__(self) -> None:
        with open(os.path.join("models", "count_vect.pkl"), 'rb') as f:
            self.count_vect = pickle.load(f)

        with open(os.path.join("models", "tf_transformer.pkl"), 'rb') as f:
            self.tf_transformer = pickle.load(f)

        with open(os.path.join("models",'token.pickle'),'rb') as f:
            self.tokenizer = pickle.load(f)
        pass

    def special(s):
        s = re.sub(r'\W', ' ',str(s))
        return s

    def single(s):
        s = re.sub(r'\s+[a-zA-Z]\s+', ' ', s)
        return s

    def singlestart(s):
        s = re.sub(r'\^[a-zA-Z]\s+', ' ', s)
        return s

    def lowercase(s):
        return s.lower()

    def mulspace(s):
        s = re.sub(r'\s+', ' ', s)
        return s

    def rt(s):
        s = re.sub(r'rt @\w+: ', ' ', s)
        return s

    def prefixedb(s):
        s = re.sub(r'^b\s+', '', s)
        return s

    def misc(s):
        s = re.sub(r'((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))|([#@]\S+)|user|\n|\t', ' ', s)
        return s

    def replace_alay(s):
        return keyword_processor.replace_keywords(s)

    def cleansing(self,s):
        s = self.special(s)
        s = self.single(s)
        s = self.singlestart(s)
        s = self.lowercase(s)
        s = self.mulspace(s)
        s = self.rt(s)
        s = self.prefixedb(s)
        s = self.misc(s)
        s = self.replace_alay(s)
        clean_text = s
        return clean_text
    
    def get_bow(self,s):
        get_bow = self.count_vect.transform(s)
        get_bow = self.tf_transformer.transform(get_bow)
        return get_bow

    def get_token(self,s):
        get_token = self.tokenizer.texts_to_sequences([s])
        get_token = pad_sequences(get_token, maxlen=128)
        return get_token