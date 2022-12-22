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

    def special(self,text):
        return re.sub(r'\W', ' ',str(text))

    def single(self,text):
        return re.sub(r'\s+[a-zA-Z]\s+', ' ', text)

    def singlestart(self,text):
        return re.sub(r'\^[a-zA-Z]\s+', ' ', text)

    def lowercase(self,text):
        return text.lower()

    def mulspace(self,text):
        return re.sub(r'\s+', ' ', text)

    def rt(self,text):
        return re.sub(r'rt @\w+: ', ' ', text)

    def prefixedb(self,text):
        return re.sub(r'^b\s+', '', text)

    def misc(self,text):
        return re.sub(r'((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))|([#@]\S+)|user|\n|\t', ' ', text)

    def replace_alay(self,text):
        return keyword_processor.replace_keywords(text)
  
    def get_bow(self,text):
        text = self.special(text)
        text = self.single(text)
        text = self.singlestart(text)
        text = self.lowercase(text)
        text = self.mulspace(text)
        text = self.rt(text)
        text = self.prefixedb(text)
        text = self.misc(text)
        text = self.replace_alay(text)
        clean_text = text
        df_ann = pd.DataFrame([text],columns=['sent_text'])
        bow = self.count_vect.transform(df_ann['sent_text'])
        bow = self.tf_transformer.transform(bow)
        return clean_text,bow

    def get_token(self,text):
        text = self.special(text)
        text = self.single(text)
        text = self.singlestart(text)
        text = self.lowercase(text)
        text = self.mulspace(text)
        text = self.rt(text)
        text = self.prefixedb(text)
        text = self.misc(text)
        text = self.replace_alay(text)
        clean_text = text
        token = self.tokenizer.texts_to_sequences([text])
        return clean_text,pad_sequences(token, maxlen=128)