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


    