from keras.models import load_model
import numpy as np
import pickle
import os

class PredictSentiment():
    def __init__(self) -> None:
        with open(os.path.join("models", "ann_model.pkl"), 'rb') as f:
            self.ann_model = pickle.load(f)
        self.lstm_model = load_model(os.path.join("models","lstm_model.h5"))       
        pass

    def ann_predict(self,bow) -> np.int64:
        result = self.ann_model.predict(bow)[0]
        return result

    def lstm_predict(self,token) -> np.int64:
        token = np.reshape(token, (1,128))
        result = self.lstm_model.predict(token,batch_size=1,verbose=0)[0]
        return np.argmax(result)
