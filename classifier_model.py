# import the necessary packages
from tensorflow.keras.layers import Activation, LeakyReLU
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model
from keras.metrics import AUC

import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from model_helper import Model_Args, Model_Helper

class Classifier():
    def __init__(self,kwargs:Model_Args):
        self.kwargs = kwargs
        self.helper = Model_Helper()
        self.loaded = False
        self.compiled = False
        self.fited = True
        self.history = None
        self.model = self.get_model(self.kwargs)        

    def get_model(self, kwargs: Model_Args):
        assert kwargs.latentDim is not None, 'latentDim must be not none'
        assert type(kwargs.latentDim) in [int,list], "latentDim must be in [int,list]"
        path = os.path.join(kwargs.model_dir,'classifier_'+kwargs.model_name+'.h5')
        if os.path.isfile(path):
            model = load_model(path)
            self.loaded = True
            self.compiled = True
            return model
        helper = self.helper
        inputs = Input(shape=kwargs.input_shape)
        x = helper.get_NN_filter(inputs,kwargs.filters,'down','leakyrelu')
        if kwargs.latent_filters is not None:
            x = helper.get_NN_filter(x,kwargs.latent_filters,'same','leakyrelu')
        x = Flatten()(x)
        if type(kwargs.latentDim) is int:
            x = Dense(kwargs.latentDim)(x)
        elif type(kwargs.latentDim) is list:
            nb_ld = len(kwargs.latentDim)-1
            for d in kwargs.latentDim[:nb_ld]:
                x = Dense(d)(x)
                x = LeakyReLU(alpha=0.2)(x)
            x = Dense(kwargs.latentDim[nb_ld])
        outputs = Activation(kwargs.activation)(x)
        model = Model(inputs,outputs=outputs,name='classifier_'+kwargs.model_name)
        return model
    
    def compile(self,loss='mse',metrics=[AUC()]):
        if self.loaded:
          return
        lr_schedule = self.kwargs.get_exp_decay()
        opt = Adam(learning_rate=lr_schedule)        
        lr_metric = self.kwargs.get_lr_metric(opt)
        metrics.append(lr_metric)
        self.model.compile(loss=loss, optimizer=opt,metrics=metrics)
        self.compiled = True
    
    def fit(self,train,val,epochs,batch_size,initial_epoch=0,es_patience=None):
        if self.compiled is False:
            self.compile()
        # Create checkpoint and earlystop
        kwargs = self.kwargs
        callbacks =[]
        checkpoint_path = os.path.join(kwargs.model_dir,'checkpoint',self.model.name)
        os.makedirs(checkpoint_path,exist_ok=True)
        filepath= os.path.join(checkpoint_path,self.model.name+"-{epoch:04d}-{val_loss:.4f}.h5")
        checkpoint = ModelCheckpoint(filepath,monitor = 'val_loss',verbose = 1, save_best_only = True,mode = 'min')
        callbacks.append(checkpoint)
        if type(es_patience) is int:
            earlystop =  EarlyStopping(monitor='val_loss', patience=es_patience,mode='min')
            callbacks.append(earlystop)
        # fit model
        H = self.model.fit(
            train,
            validation_data=val,
            epochs=epochs,
            batch_size=batch_size,
            initial_epoch=initial_epoch,
            callbacks=callbacks)
        self.history = H.history        
        self.epochs = epochs
        self.batch_size = batch_size
        self.fited = True
    
    def save(self):
        save_dir = self.kwargs.model_dir
        assert self.compiled and self.fited, 'Must compile and fit'
        ae_path = os.path.join(save_dir,self.model.name+'.h5')        
        h_path = os.path.join(save_dir,self.model.name+'.csv')
        self.model.save(ae_path)        
        df = pd.DataFrame(self.history)
        df.to_csv(h_path)
    
    def plot_show(self):
        if self.loaded and self.history is None:
          history_path = os.path.join(self.kwargs.model_dir,self.model.name+'.csv')
          self.history = pd.read_csv(history_path)
        assert self.history is not None, 'You must fit model before run this'
        # construct a plot that plots and saves the training history
        N = np.arange(0, len(self.history))
        plt.style.use("ggplot")
        plt.figure()
        plt.plot(N, self.history["loss"], label="train_loss")
        plt.plot(N, self.history["val_loss"], label="val_loss")
        plt.title("Training Loss and Accuracy")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss/Accuracy")
        plt.legend(loc="lower left")
        plt.show()
        