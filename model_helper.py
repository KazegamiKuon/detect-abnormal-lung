import tensorflow as tf
from keras.optimizers.schedules import ExponentialDecay

class Model_Args():
    def __init__(self,
                input_shape,
                activation='sigmoid',
                activation_all=False,
                filters=(32,64),
                latent_filters=None,
                latentDim=256,
                model_name='',
                model_dir='',
                decay_steps=100000,
                decay_rate=0.96,
                lr = 1e-3,
                decay_name = 'exp_decay'):
        '''
        params:
            input_shape: size của ảnh đầu vào
            avtivation: Hàm kích hoạt của đầu ra, mặc định là sigmoid
            filters: số filter từng lớp, mặc định là (32,64)
            lattenDim: chiều của lớp giữa (Dense), mặc định là 256. Nếu là none thì mô hình sẽ ko tạo dense latent
            decay_name: tên của mô hình
        '''
        self.input_shape = input_shape
        self.activation=activation
        self.activation_all=activation_all
        self.filters=filters
        self.latent_filters=latent_filters
        self.latentDim=latentDim
        self.model_name=model_name
        self.model_dir=model_dir
        self.decay_steps=decay_steps
        self.decay_rate=decay_rate
        self.lr = lr
        self.decay_name = decay_name
    
    def get_lr_metric(self,optimizer):
        def lr(y_true, y_pred):
            return optimizer._decayed_lr(tf.float32) # I use ._decayed_lr method instead of .lr
        return lr
    
    def get_exp_decay(self):
        '''
        params:
            start_decay: dùng để cài đặt lr bắt đầu trong trường hợp khôi phục lại training
        '''
        return ExponentialDecay(self.lr,self.decay_steps,self.decay_rate,name=self.decay_name)

from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D, Conv2DTranspose
from tensorflow.keras.layers import ReLU, LeakyReLU

class Model_Helper():
    def __init__(self):
        pass
    def get_Conv2D_down(self,x,f):
        return Conv2D(f,(3,3),strides=2,padding='same')(x)

    def get_Conv2D_up(self,x,f):
        return Conv2DTranspose(f,(3,3),strides=2,padding='same')(x)

    def get_Conv2D_same(self,x,f):
        return Conv2D(f,(3,3),strides=1,padding='same')(x)

    def get_NN_filter(self,start_x,filters,scale,activation):
        assert scale in ['down','up','same'], "scale must be in ['down','up','same']"
        assert activation in ['relu','leakyrelu'], "activation must be in ['relu','leakyrelu']"
        x = start_x
        chanDim = -1
        get_conv = self.get_Conv2D_down if scale is 'down' else self.get_Conv2D_up if scale is 'up' else self.get_Conv2D_same
        for f in filters:
            x = get_conv(x,f)
            if activation is 'relu':
                x = ReLU(max_value=255)(x)
            elif activation is 'leakyrelu':
                x = LeakyReLU(alpha=0.2)(x)
            x = BatchNormalization(axis=chanDim)(x)
        return x