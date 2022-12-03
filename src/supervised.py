import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation

class supervised_model():
    def __init__(self, model):
        '''
        Takes an inputs model
        '''
        self.model = model 

    def train(self, x_train , y_train):
        '''
        train the model
        '''
        self.model.fit(x_train , y_train)

    def predict(self, x_test):
        '''
        make prediction and prediction prob on the data
        '''
        y_pred = self.model.predict(x_test) 
        y_pred_proba = self.model.predict_proba(x_test)[:,1]
        return y_pred , y_pred_proba

class FeedForwardNN(object):
    def __init__(self,in_dims):
        super().__init__()

        self.in_dims = in_dims

    def create_net(self):
        model = Sequential()
        model.add(tf.keras.Input((self.in_dims)))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))

        return model
    
    def compile_net(self, model):
        self.model = model
        self.model.compile(loss='binary_crossentropy',
              optimizer="adam",
              metrics=[tf.keras.metrics.AUC()])
        return self.model