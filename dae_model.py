# import the necessary packages
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D, UpSampling2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import ReLU, LeakyReLU
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
import numpy as np
import os
import pandas as pd
from keras.models import load_model
import tensorflow as tf

from tensorflow.keras.optimizers import Adam
from keras.optimizers.schedules import ExponentialDecay
from keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt

class DAE_Model():
    def __init__(self,input_shape, **kwargs):
        '''
        params:
            input_shape: size của ảnh đầu vào
            avtivation: Hàm kích hoạt của đầu ra, mặc định là sigmoid
            filters: số filter từng lớp, mặc định là (32,64)
            latten_filters: số filter lớp giữa mà ở đó không bị giảm scale
            lattenDim: chiều của lớp giữa (Dense), mặc định là 256. Nếu là none thì lớp giữa là output của encoder
            model_name: tên của mô hình
            model_dir: thư mục lưu model
        '''
        # print(kwargs)
        # (encoder, decoder, autoencoder) = self.dae_build(input_shape,**kwargs)
        (encoder, decoder, autoencoder) = self.dae_build(input_shape,**kwargs)
        self.autoencoder = autoencoder
        self.encoder = encoder
        self.decoder = decoder
        self.loaded = encoder is None and decoder is None and autoencoder is not None
        # self.autoencoder = None
        # self.encoder = None
        # self.decoder = None
        # self.input_shape = input_shape
        # self.dae_kwargs = kwargs
        self.model_dir = kwargs['model_dir']
        self.history = None
        self.epochs = None
        self.batch_size = None
        self.compiled = self.loaded
        self.fited = False
        self.decay_steps = 100000
    def set_decay_steps(self,decay_steps):
        self.decay_steps = decay_steps
    
    # def resume_model(self,model_path,train,val,epochs,batch_size,initial_epoch,es_patience):
    #     self.loaded = False
    #     self.autoencoder = load_model(model_path,compile=False)
    #     self.compile()
    #     self.loaded = True
    #     self.fit(train,val,epochs,batch_size,initial_epoch=initial_epoch,es_patience=es_patience)
    
    def __get_lr_metric(self,optimizer):
        def lr(y_true, y_pred):
            return optimizer._decayed_lr(tf.float32) # I use ._decayed_lr method instead of .lr
        return lr
    def __get_ssim_loss(self):
        def ssim_loss(y_true, y_pred):
            return tf.reduce_mean(tf.image.ssim(y_true, y_pred,max_val=255))
        return ssim_loss
    def compile(self,loss='mse'):
        if self.loaded:
          return        
        lr_schedule = ExponentialDecay(1e-3,self.decay_steps,0.96,name='exp_decay')
        opt = Adam(learning_rate=lr_schedule)
        metrics = []
        lr_metric = self.__get_lr_metric(opt)
        metrics.append(lr_metric)
        ssim_loss = self.__get_ssim_loss()
        metrics.append(ssim_loss)
        self.autoencoder.compile(loss=loss, optimizer=opt,metrics=metrics)
        self.compiled = True
    
    def fit(self,train,val,epochs,batch_size,initial_epoch=0,es_patience=None):
        if self.compiled is False:
            self.compile()
        # Create checkpoint and earlystop
        callbacks =[]
        checkpoint_path = os.path.join(self.model_dir,'checkpoint',self.autoencoder.name)
        os.makedirs(checkpoint_path,exist_ok=True)
        filepath= os.path.join(checkpoint_path,self.autoencoder.name+"-{epoch:04d}-{val_loss:.4f}.h5")
        checkpoint = ModelCheckpoint(filepath,monitor = 'val_loss',verbose = 1, save_best_only = True,mode = 'min')                
        callbacks.append(checkpoint)
        if type(es_patience) is int:
            earlystop =  EarlyStopping(monitor='val_loss', patience=es_patience,mode='min')
            callbacks.append(earlystop)
        # fit model
        H = self.autoencoder.fit(
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
        save_dir = self.model_dir
        assert self.compiled and self.fited, 'Must compile and fit'
        ae_path = os.path.join(save_dir,self.autoencoder.name+'.h5')        
        h_path = os.path.join(save_dir,self.autoencoder.name+'.csv')
        self.autoencoder.save(ae_path)
        if self.encoder is not None:
            e_path = os.path.join(save_dir,self.encoder.name+'.h5')
            self.encoder.save(e_path)
        if self.decoder is not None:
            d_path = os.path.join(save_dir,self.decoder.name+'.h5')
            self.decoder.save(d_path)
        df = pd.DataFrame(self.history)
        df.to_csv(h_path)
    
    def plot_show(self):
        if self.loaded and self.history is None:
          history_path = os.path.join(self.model_dir,self.autoencoder.name+'.csv')
          self.history = pd.read_csv(history_path)
          self.epochs = len(self.history)
        assert self.history is not None, 'You must fit model before run this'
        # construct a plot that plots and saves the training history
        N = np.arange(0, len(self.history['loss']))
        plt.style.use("ggplot")
        plt.figure()
        plt.plot(N, self.history["loss"], label="train_loss")
        plt.plot(N, self.history["val_loss"], label="val_loss")
        plt.title("Training Loss and Accuracy")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss/Accuracy")
        plt.legend(loc="lower left")
        plt.show()

    def dae_build(self,input_shape,activation='sigmoid',activation_all=False,filters=(32,64),latent_filters=None,latentDim=256,model_name='',model_dir=''):
        '''
        params:
            input_shape: size của ảnh đầu vào
            avtivation: Hàm kích hoạt của đầu ra, mặc định là sigmoid
            filters: số filter từng lớp, mặc định là (32,64)
            lattenDim: chiều của lớp giữa (Dense), mặc định là 256. Nếu là none thì lớp giữa là output của encoder
            name: tên của mô hình
        return:
            trả về 3 mô hình: encoder, decoder, autoencoder
        '''
        ae_path = os.path.join(model_dir,'autoencoder_'+model_name+'.h5')
        if os.path.isfile(ae_path):
            autoencoder = load_model(ae_path)
            return None, None, autoencoder
        (height, width, depth) = input_shape
        # start building encoder inputs same input autoencoder
        inputs = Input(shape=input_shape)
        chanDim = -1
        x = inputs
        #loop over the number filters
        for f in filters:
            # CONV => RELU => BN operation
            # scale down
            x = Conv2D(f,(3,3),padding='same')(x)
            if activation_all:
              x = ReLU(max_value=255)(x)
            else:
              x = LeakyReLU(alpha=0.2)(x)
            x = BatchNormalization(axis=chanDim)(x)
            x = MaxPooling2D()(x)
        if latent_filters is not None:
            for f in latent_filters:
                x = Conv2D(f,(3,3),strides=1,padding='same')(x)
                if activation_all:
                  x = ReLU(max_value=255)(x)
                else:
                  x = LeakyReLU(alpha=0.2)(x)
                x = BatchNormalization(axis=chanDim)(x)
        volumeSize = K.int_shape(x)
        latent = x
        if latentDim is not None:
            x = Flatten()(x)
            latent = Dense(latentDim)(x)
        encoder = Model(inputs=inputs,outputs=latent,name='encoder_'+model_name)
        print(encoder.output_shape[1:])
        # start building the decoder use output encoder as inputs
        latentInputs = Input(shape=encoder.output_shape[1:])
        x = latentInputs
        if latentDim is not None:
            x = Dense(np.prod(volumeSize[1:]))(x)
            x = Reshape((volumeSize[1],volumeSize[2],volumeSize[3]))(x)
        if latent_filters is not None:
            for f in latent_filters[::-1]:
                x = Conv2D(f,(3,3),strides=1,padding='same')(x)
                if activation_all:
                  x = ReLU(max_value=255)(x)
                else:
                  x = LeakyReLU(alpha=0.2)(x)
                x = BatchNormalization(axis=chanDim)(x)
        #loop filters but reverse order
        for f in filters[::-1]:
            # CONV_TRANSPOSE => RELU => BN operation
            x = Conv2DTranspose(f,(3,3),strides=2,padding="same")(x)
            if activation_all:
              x = ReLU(max_value=255)(x)
            else:
              x = LeakyReLU(alpha=0.2)(x)
            x = BatchNormalization(axis=chanDim)(x)
            # CONV => RELU => BN operation
            x = Conv2D(f,(3,3),strides=1,padding='same')(x)
            if activation_all:
              x = ReLU(max_value=255)(x)
            else:
              x = LeakyReLU(alpha=0.2)(x)
            x = BatchNormalization(axis=chanDim)(x)
        x = Conv2DTranspose(depth,(3,3),padding='same')(x)
        outputs = Activation(activation)(x)
        #decoder model
        decoder = Model(latentInputs,outputs=outputs,name='decoder_'+model_name)
        # Autoencoder is encoder + decoder
        autoencoder = Model(inputs,decoder(encoder(inputs)),name='autoencoder_'+model_name)
        # return 3 model encoder, decoder, autoencoder
        return (encoder, decoder, autoencoder)
    