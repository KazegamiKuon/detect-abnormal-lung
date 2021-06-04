import cv2
import re
import numpy as np
import pandas as pd
import os
import imgaug.augmenters as iaa
from tqdm.auto import tqdm
from tensorflow.keras.utils import Sequence, to_categorical
from tensorflow.image import ssim

class ProcessData():
    def __init__(self,seed=42):
        self.seed = seed

    def iou_score(self,ground_truth,predict_data):
        '''
        params:
          ground_truth: vùng mốc để so sánh
          predict_data: Dữ liệu được predict
        return:
          IOU score
        '''
        intersection = np.logical_and(ground_truth, predict_data)
        union = np.logical_or(ground_truth, predict_data)
        iou_score = np.sum(intersection) / np.sum(union)
        return iou_score
    
    def get_mean_ssim(xs,ys):
        '''
        params:
          xs,ys: danh sách hình cần được so sánh
        return:
          giá trị trung bình của độ tương đồng 2 danh sách
        '''
        nb = len(xs)
        ssims = []
        for i in tqdm(range(nb)):
          s = ssim(xs[i].astype(float),ys[i].astype(float),255)
          ssims.append(s.numpy())
        ssims = np.array(ssims)
        return ssims.sum()/nb

    def get_img_names(self,path):
        '''
        params:
          path: đường dẫn file csv
        return:
          danh sách tên của hình
        '''
        return pd.read_csv(path)['image_id'].unique()
    
    def get_paths(self,path_dict,tail):
        '''
        params:
          path_dict: type dictionary. key là load path, value là filename được load trong đó
          tail: đuôi của thư mục
        return:
          paths = [] danh sách đường dẫn
        '''
        paths = []
        for key, values in path_dict.items():
            paths.extend([os.path.join(key,value+tail) for value in values])
        return np.array(paths)
    def resize_img(self,img,size):
        '''
        params:
          img: ảnh
          size: size ảnh cần resize về
        return:
          ảnh đã được resize
        '''
        return cv2.resize(img,size)
    
    def get_img(self,path,size=None,gray=True):
        '''
        params:
          path: path của hình
          size: resize hình về size, mặc định ko resize
          gray: load hình gray hoặc không, mặc định True
        return:
          kiểu dữ liệu giống với kiểu được opencv.imread trả về
        '''
        img = np.expand_dims(cv2.imread(path,cv2.IMREAD_GRAYSCALE),axis=-1) if gray else cv2.imread(path)
        real_shape = img.shape
        if size is not None:
            img = np.expand_dims(self.resize_img(img,size),axis=-1) if gray else self.resize_img(img,size)
        return real_shape,img
    
    def get_imgs(self,paths,show_process=True,**kwargs):
        '''
        params:
          paths: danh sách path của hình
          size: resize hình về size, mặc định ko resize
          gray: load hình gray hoặc không, mặc định True
        return:
          danh sách những hình có kiểu dữ liệu giống với kiểu được opencv.imread trả về
        '''
        shapes, imgs = self.get_shapes_and_imgs(paths,show_process,**kwargs)
        return imgs
    
    def get_shapes_and_imgs(self,paths,show_process=True,**kwargs):
        '''
        params:
          paths: danh sách path của hình
          size: resize hình về size, mặc định ko resize
          gray: load hình gray hoặc không, mặc định True
        return:
          danh sách những hình có kiểu dữ liệu giống với kiểu được opencv.imread trả về
        '''
        s_process = lambda x,desc=None : x
        if show_process is True:
          s_process = tqdm
        imgs =[]
        shapes = []
        for path in s_process(paths,desc='load_img'):
            shape, img = self.get_img(path,**kwargs)
            imgs.append(img)
            shapes.append(shape)
        shapes = np.array(shapes)
        return shapes, imgs

    def get_mask(self,path,size=None):
        '''
        params:
          path: đường dẫn của mask
          size: resize mask về size, mặc định ko resize
        return:
          mask: np.ndarray
        '''
        mask = np.load(path)
        if size is not None:
            mask = np.expand_dims(cv2.resize(mask,size),axis=-1)
        if len(mask.shape) < 3:
            mask = np.expand_dims(mask,axis=-1)
        return mask.astype(np.uint8)

    def get_masks(self,paths,show_process=True,**kwargs):
        '''
        params:
          path: đường dẫn của mask
          size: resize mask về size, mặc định ko resize
        return:
          mask_size, masks: size, danh sách mask
        '''
        s_process = lambda x,desc=None : x
        if show_process is True:
          s_process = tqdm
        masks =[]
        for path in s_process(paths,desc='load_mask'):
            mask = self.get_mask(path,**kwargs)
            masks.append(mask)
        masks = np.array(masks)
        return masks.shape[1:3], masks

    def get_npy(self,path,size=None):
        '''
        params:
          path: đường dẫn của mask
          size: resize mask về size, mặc định ko resize
        return:
          mask: np.ndarray
        '''
        npy = np.load(path)
        if size is not None:
            npy = np.expand_dims(cv2.resize(npy,size),axis=-1)
        return npy

    def get_npies(self,paths,**kwargs):
        '''
        params:
          path: đường dẫn của mask
          size: resize mask về size, mặc định ko resize
        return:
          mask_size, masks: size, danh sách mask
        '''
        npies =[]
        for path in tqdm(paths,desc='load_mask'):
            npy = self.get_npy(path,**kwargs)
            npies.append(npy)
        npies = np.array(npies)
        return npies.shape[1:3], npies

    def save_masks(self,img_names,masks,save_dir):
        '''
        params:
          img_names: Tên của những mask
          masks: data về mask
          save_dir: thư mục lưu
        '''
        os.makedirs(save_dir,exist_ok=True)
        for i in tqdm(range(len(img_names)),desc='save_mask'):
            path = os.path.join(save_dir,img_names[i])
            np.save(path,masks[i])

    def get_data(self,imgs,masks):
        return imgs*masks
    
    def get_aug_snow_line(self,my_seed = None):
        seed = self.seed
        if my_seed is not None:
            seed = my_seed
        return iaa.weather.Snowflakes(density=[0.035,0.04],density_uniformity=0.5,flake_size=0.7,flake_size_uniformity=0.7,speed=0.1,angle=[80,115],random_state=seed)
    
    def get_aug_snow_big(self,my_seed = None):
        seed = self.seed
        if my_seed is not None:
            seed = my_seed
        return iaa.weather.Snowflakes(density=0.07,density_uniformity=0.5,flake_size=0.98,flake_size_uniformity=0.7,speed=[0.0,0.05],angle=[80,115],random_state=seed)

    def get_aug_drop_out(self,my_seed = None):
        seed = self.seed
        if my_seed is not None:
            seed = my_seed
        return iaa.CoarseDropout(p=(0.3,0.4),size_percent=0.009,random_state=seed)

    def get_aug_someof(self,apply,augs,my_seed = None):
        '''
        params:
          apply: số lượng apply aug
          augs: danh sách aug
          my_seed: random_state
        return:
          aug
        '''
        seed = self.seed
        if my_seed is not None:
            seed = my_seed
        return iaa.SomeOf(apply,augs,random_state=seed)

    def get_aug_oneof(self,augs,my_seed = None):
        '''
        params:
          augs: danh sách aug
          my_seed: random_state
        return:
          aug
        '''
        seed = self.seed
        if my_seed is not None:
            seed = my_seed
        return iaa.OneOf(augs,random_state=seed)
    
    def get_data_noise(self,data,aug,masks=None,show_process=True):
        '''
        params:
          data: danh sách data
          masks: những mask apply vô data
          aug: agument object
        return:
          aug
        '''
        s_process = lambda x,desc=None : x
        if show_process is True:
          s_process = tqdm
        img_aug = []
        for d in s_process(data,desc='create noise'):
            img_aug.append(aug(image=d))
        img_aug = np.array(img_aug)[:,:,:,:data.shape[3]]
        if masks is not None:
          img_aug = img_aug*masks
        return img_aug
    
    def get_aug_by_type(self,ntype,my_seed=None):
        seed = self.seed
        if my_seed is not None:
            seed=my_seed
        #Tạo noise dạng bông tuyết
        snowflake_big = iaa.weather.Snowflakes(density=0.07,density_uniformity=0.5,flake_size=0.98,flake_size_uniformity=0.7,speed=[0.0,0.05],angle=[80,115],random_state=seed)
        snowflake_line = iaa.weather.Snowflakes(density=[0.035,0.04],density_uniformity=0.5,flake_size=0.7,flake_size_uniformity=0.7,speed=0.1,angle=[80,115],random_state=seed)
        #P1: chọn aplly 1 hoặc 2 cách trên
        someof = iaa.SomeOf((1,2),[snowflake_big,snowflake_line],random_state=seed)
        
        #P2: Tạo noise dropout hình
        codrop = iaa.CoarseDropout(p=(0.3,0.4),size_percent=0.006,random_state=seed)

        aug_list = []
        if ntype == 3:
            aug_list.append(codrop)
            aug_list.append(someof)
        elif ntype == 2:
            aug_list.append(someof)
        elif ntype == 1:
            aug_list.append(codrop)
        else:
            assert False, "ntype is one of: 1 - drop_out, 2 - snow, 3 - all"
        #chọn 1 trong 2 P1 hoặc P2 
        aug = iaa.OneOf(aug_list,random_state=seed)
        return aug
    
    def get_data_noise_and_save(self,data_paths,mask_paths,noise_paths,ntype=3,my_seed = None):
        aug = self.get_aug_by_type(ntype,my_seed)

        for i in tqdm(range(len(noise_paths))):
            noise_path = noise_paths[i]
            if os.path.isfile(noise_path) is False:
              img = self.get_img(data_paths[i])
              mask = self.get_mask(mask_paths[i]).astype(np.uint8)
              noise = aug(image=img)[:,:,:1].astype(np.uint8)
              noise = noise*mask
              cv2.imwrite(noise_path,noise)

    def get_data_noise_vlazy(self,data,masks=None,ntype=3,my_seed=None):
        aug = self.get_aug_by_type(ntype,my_seed)
        img_aug = []
        for d in tqdm(data,desc='noise_type_{:02d}'.format(ntype)):
            img = aug(image=d.astype(np.uint8)).astype(np.uint8)
            img_aug.append(img[:,:,:1])
        img_aug = np.array(img_aug)
        if masks is not None:
          img_aug = img_aug*masks
        return img_aug

    def recover_mask(self,model,best_mean,masks,batch_size):
        remasks = model.predict(masks,batch_size=batch_size)
        remasks[remasks < best_mean] = 0
        remasks[remasks >= best_mean] = 1
        return remasks
  
    def recover_lung(self,model,lungs,masks,batch_size):
        relungs = model.predict(lungs,batch_size=batch_size)
        relungs = pdata.get_data(relungs,masks)
        return relungs

    def get_best_mean(self,path):
        bm_path = path.replace('.h5','_best_mean.csv')
        data = pd.read_csv(bm_path)
        index = data['score'].values.argmax()
        return data['mean'][index]

class DataGenerator(Sequence):
    'Generates data for Keras'
    def __init__(self,
                 x,
                 y,                 
                 batch_size,
                 shuffle=True,
                 preprocessor=None):
        '''
        x: danh sách x
        y: danh sách y
        preprocessor: Hàm nhận aguments là x, y trả về x, y đã biến đổi
        batch_size: kích thước của 1 batch        
        shuffle: có shuffle dữ liệu sau mỗi epoch hay không?
        '''
        assert isinstance(x,(list,np.ndarray)), 'x must be list or np.ndarray'
        assert isinstance(y,(list,np.ndarray)), 'y must be list or np.ndarray'
        assert len(x) == len(y), 'x must same size as y'
        self.x = x
        self.y = y
        self.preprocessor = preprocessor
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        '''
        return:
          Trả về số lượng batch/1 epoch
        '''
        return int(np.floor(len(self.x) / self.batch_size))
    
    def get_nb_steps(self):
        return self.__len__()

    def __getitem__(self, index):
        '''
        params:
          index: index của batch
        return:
          X, y cho batch thứ index
        '''
        # Lấy ra indexes của batch thứ index
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # print('indexes',indexes)
        # List all_filenames trong một batch
        x_temp, y_temp = ([self.x[k] for k in indexes],[self.y[k] for k in indexes])

        # Khởi tạo data
        x, y = self.__data_generation(x_temp,y_temp)

        return x, y

    def on_epoch_end(self):
        '''
        Shuffle dữ liệu khi epochs end hoặc start.
        '''
        self.indexes = np.arange(len(self.x))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, x_temp,y_temp):
        '''
        params:
          all_filenames_temp: list các filenames trong 1 batch
        return:
          Trả về giá trị cho một batch.
        '''
        x, y = (x_temp, y_temp) if self.preprocessor is None else self.preprocessor(x_temp,y_temp)
        x = np.array(x)
        y = np.array(y)
        return x, y