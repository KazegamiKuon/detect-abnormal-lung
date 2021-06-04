from process_data import ProcessData
from keras.models import load_model
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import cv2
import types

class IType():
  def __init__(self):
    self.src = 1
    self.lung = 2
    self.abnormal = 3

class XRayObject():
  def __init__(self,path,img,real_size,mask,remask,lung,relung,abnormalU,abnormalD):
    self.path = path
    self.img = img
    self.real_size = real_size
    self.mask = mask
    self.remask = remask
    self.lung = lung
    self.relung = relung
    self.abnormalU = abnormalU
    self.abnormalD = abnormalD
    self.__s__ = 400
    # Biến kích hoạt, nếu ở trạng thái false thì các giá trị bên dưới đều là None
    self.__processed__ = False
    self.blur_abnormal = None
    self.all_boxs = None
    self.boxs = None
    # Hằng:
    self.__x__ = 0
    self.__y__ = 1
    self.__w__ = 2
    self.__h__ = 3
    self.process()

  def process(self):    
    # Phân cực dữ liệu (threshold)
    threshabU = cv2.threshold(self.abnormalU[:,:,0],0,255,cv2.THRESH_BINARY)[1]
    threshabD = cv2.threshold(self.abnormalD[:,:,0],0,255,cv2.THRESH_BINARY)[1]
    # Giảm nhiễu (smoth median)
    blurabU = cv2.medianBlur(threshabU,9)
    blurabD = cv2.medianBlur(threshabD,9)
    self.blur_abnormal = cv2.threshold(blurabU + blurabD,0,255,cv2.THRESH_BINARY)[1].astype(np.uint8)
    self.all_boxs = self.get_boxs(self.blur_abnormal)
    self.boxs = self.get_boxs_by_s(self.all_boxs,self.__s__)
  
  def get_boxs(self,img):    
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    boxs = np.array([cv2.boundingRect(contour) for contour in contours])
    return boxs
  
  def get_boxs_by_s(self,boxs,s):
    rboxs = []    
    for box in boxs:
      if box[self.__w__]*box[self.__h__] >= s:
        rboxs.append(box)
    return np.array(rboxs)
  
  def get_boxs_real_size(self,boxs):
    img = self.img
    coef_x = self.real_size[0]/img.shape[0]
    coef_y = self.real_size[1]/img.shape[1]
    rboxs = []
    for box in boxs:
      x = int(box[self.__x__]*coef_x)
      w = int(box[self.__w__]*coef_x)
      y = int(box[self.__y__]*coef_y)
      h = int(box[self.__h__]*coef_y)
      rboxs.append((x,y,w,h))
    return np.array(rboxs)
  
  def get_view_object(self,itype,real_size=False,all_boxs = False):
    # Biến loại hình
    imgt = IType()
    # scale
    coef_x = 1
    coef_y = 1
    img = self.img
    # nếu loại ảnh là phổi thì img là lung
    if itype == imgt.lung:
      img = self.lung[:,:,0]
    # nếu loại ảnh là abnormal thì img là blur_abnormal
    if itype == imgt.abnormal:
      img = self.blur_abnormal
    # nếu show ảnh kích thước thật. Resize và cập nhật scale
    if real_size:
      coef_x = self.real_size[0]/img.shape[0]
      coef_y = self.real_size[1]/img.shape[1]
      img = cv2.resize(img,self.real_size)
    # nếu không phải ảnh gốc (3chanels) thì đổi sang ảnh 3 chanel
    if itype is not imgt.src:
      img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
    # box
    boxs = self.boxs
    # nếu vẽ tất cả box
    if all_boxs:
      boxs = self.all_boxs    
    for box in boxs:
      x = int(box[self.__x__]*coef_x)
      w = int(box[self.__w__]*coef_x)
      y = int(box[self.__y__]*coef_y)
      h = int(box[self.__h__]*coef_y)
      cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3)
    return img

  def view_object(self,figsize=(20,10),itype = IType().src):
    img = self.get_view_object(itype,real_size=True)
    plt.figure(figsize=figsize)
    plt.imshow(img)
    plt.show()

class AbnormalDetector():
  def __init__(self,mmodel_path,rmmodel_paths,rlmodel_path,img_paths,batch_size,my_seed=42,smooth=None,run_in_smooth=False,rlb_mean=None):
    pdata = ProcessData(seed=my_seed)
    self.mmodel = load_model(mmodel_path,compile=False)
    self.rmmodels = [load_model(rmmodel_path,compile=False) for rmmodel_path in rmmodel_paths]
    self.rlmodel = load_model(rlmodel_path,compile=False)
    self.rmb_means = [pdata.get_best_mean(rmmodel_path) for rmmodel_path in rmmodel_paths]
    self.rlb_mean = pdata.get_best_mean(rlmodel_path)
    if rlb_mean is not None:
      self.rlb_mean = rlb_mean
    self.pdata = pdata
    self.xrayobjects = []
    self.__process__(img_paths,batch_size,smooth,run_in_smooth)

  def get_mask_path(self,img_path):
    return img_path.replace('png','mask',1)

  def recover_masks(self,masks,batch_size):
    pdata = self.pdata
    # Lấy model khôi phục mask
    models = self.rmmodels
    num_model = len(models)    
    #lấy chuẩn mean của model
    bmeans = self.rmb_means
    remasks = masks.copy()
    # Khôi phục mask
    for i in range(num_model):
      models[i].predict(remasks,batch_size,verbose=1)
      # smoothing mask
      remasks = np.array([cv2.blur(remask,(5,5)) for remask in remasks],dtype=np.uint8)
      remasks = np.expand_dims(remasks,axis=-1)
      # Chuẩn hóa mask
      remasks[remasks < bmeans[0]] = 0
      remasks[remasks >= bmeans[0]] = 1      
      remasks = np.array([cv2.medianBlur(remask,23) for remask in remasks],dtype=np.uint8)
      remasks = np.expand_dims(remasks,axis=-1)
    return remasks
  
  def recover_lungs(self,lungs,batch_size):
    pdata = self.pdata
    # Lấy model khôi phục lung
    model = self.rlmodel
    # Khôi phục lung
    relungs = model.predict(lungs,batch_size,verbose=1)
    return relungs

  def get_abnormals(self,lungs,relungs,rlb_mean=None,ctype='*'):
    # lấy chuẩn để so sánh. Đây là độ lệch trung bình của hình predict với hình gốc
    bmean = self.rlb_mean
    if rlb_mean is not None:
      bmean = rlb_mean
    tempU = relungs.copy()
    tempD = relungs.copy()
    if ctype is '*':
      tempU = relungs*(1+bmean)
      tempD = relungs*(1-bmean)
    elif ctype is '+':
      tempU = relungs+bmean
      tempD = relungs-bmean
    
    abnormalU = np.zeros(lungs.shape)
    abnormalU[lungs >= tempU] = lungs[lungs >= tempU]

    abnormalD = np.zeros(lungs.shape)
    abnormalD[lungs <= tempD] = lungs[lungs <= tempD]
    return abnormalU.astype(np.uint8), abnormalD.astype(np.uint8)
  
  def update_abnormals(self,rlb_mean,ctype='*'):
    temps = []
    for xobject in tqdm(self.xrayobjects):
      tabU, tabD = self.get_abnormals(np.array([xobject.lung]),np.array([xobject.relung]),rlb_mean,ctype)
      xobject.abnormalU = tabU[0]
      xobject.abnormalD = tabD[0]
      xobject.process()

  def view_objects(self,figsize=(20,10),itype = IType().src):
    for xobject in self.xrayobjects:
      xobject.view_object(figsize=figsize,itype=itype)

  def __process__(self,img_paths,bs,smooth,run_in_smooth):
    input_size = self.mmodel.input_shape[1:3]
    pdata = self.pdata
    rshapes, imgs = pdata.get_shapes_and_imgs(img_paths,size=input_size,gray=False)
    imgs = np.array(imgs)
    ismooths = imgs.copy()
    if isinstance(smooth,types.FunctionType):
      ismooths = np.array([smooth(ismooth) for ismooth in ismooths],dtype=np.uint8)
    print('segment mask:')
    masks = self.mmodel.predict(ismooths,bs,verbose=1)
    thr = 0.2
    masks[masks >= thr] = 1
    masks[masks < thr] = 0
    print('recover mask:')
    remasks = self.recover_masks(masks,bs)
    lungs = remasks*imgs[:,:,:,:1]
    if isinstance(smooth,types.FunctionType) and run_in_smooth:
      lungs = np.expand_dims(np.array([smooth(lung) for lung in lungs]),axis=-1)
    print('recover lung:')
    relungs = self.recover_lungs(lungs,bs)
    relungs = relungs*remasks
    if isinstance(smooth,types.FunctionType) and run_in_smooth:
      lungs = lungs*remasks
    abnormalU, abnormalD = self.get_abnormals(lungs,relungs)
    nb_img = len(img_paths)
    self.xrayobjects = [XRayObject(img_paths[i], #đường hình
                                   imgs[i], #hình được load lên
                                   (rshapes[i][0], rshapes[i][1]), #kích cỡ thật
                                   masks[i], # Mặt lạ phổi
                                   remasks[i], # Mặt lạ phổi được khôi phục
                                   lungs[i], # Phổi
                                   relungs[i], #Phổi trạng thái bình thường
                                   abnormalU[i], # Bất thường trên
                                   abnormalD[i] # Bất thường dưới
                                   ) for i in tqdm(range(nb_img),desc='Tạo XObject')]