import matplotlib.pyplot as plt
import numpy as np

class XABViewer():
    def __init__(self,model,normal,abnormal):
        self.normal = normal
        self.abnormal = abnormal
        self.pred = model.predict(abnormal)
        self.ab_nor = abnormal - normal
        self.pred_nor = self.pred - normal
        self.ab_pred = abnormal - self.pred
        self.abn_abp = self.ab_nor - self.ab_pred

    def view_lung(self,nb_img):
        plt.style.use("default")
        plt.figure(figsize=(20,10))
        nb_row = 1
        nb_col = 4

        plt.subplot(nb_row,nb_col,1,xlabel='Lung Normal')
        plt.imshow(self.normal[nb_img, :,:,0],cmap='gray')

        plt.subplot(nb_row,nb_col,2,xlabel='Lung Abnormal')
        plt.imshow(self.abnormal[nb_img, :,:,0],cmap='gray')

        plt.subplot(nb_row,nb_col,3,xlabel='Lung Predict')
        plt.imshow(self.pred[nb_img, :,:,0],cmap='gray')

        plt.subplot(nb_row,nb_col,4,xlabel='Abnormal')
        plt.imshow(self.ab_pred[nb_img, :,:,0],cmap='gray')
        plt.show()

    def view_compare_normal(self,nb_img):
        plt.style.use("default")
        plt.figure(figsize=(20,10))
        nb_row = 1
        nb_col = 3

        plt.subplot(nb_row,nb_col,1,xlabel='Lung Normal')
        plt.imshow(self.normal[nb_img, :,:,0],cmap='gray')

        plt.subplot(nb_row,nb_col,2,xlabel='Lung Predict')
        plt.imshow(self.pred[nb_img, :,:,0],cmap='gray')

        plt.subplot(nb_row,nb_col,3,xlabel='Normal - Predict')
        plt.imshow(self.pred_nor[nb_img, :,:,0],cmap='gray')
        plt.show()

    def view_compare_abnormal(self,nb_img):
        plt.style.use("default")
        plt.figure(figsize=(20,10))
        nb_row = 1
        nb_col = 3

        plt.subplot(nb_row,nb_col,1,xlabel='Lung Abnormal - Normal')
        plt.imshow(self.ab_nor[nb_img, :,:,0],cmap='gray')

        plt.subplot(nb_row,nb_col,2,xlabel='Lung Abnormal - Predict')
        plt.imshow(self.ab_pred[nb_img, :,:,0],cmap='gray')
        
        plt.subplot(nb_row,nb_col,3,xlabel='Lung Abnormal - Normal - Abnormal - Predict')
        plt.imshow(self.abn_abp[nb_img, :,:,0],cmap='gray')
        plt.show()

    def view_plot(self,nb_img,yaxis):
        plt.style.use("ggplot")
        plt.figure(figsize=(100,10))
        nb_row = 1
        nb_col = 4

        size = self.pred.shape[2]
        xaxis = np.arange(0,size)
        plt.subplot(nb_row,nb_col,1,xlabel='Lung Normal')
        plt.plot(xaxis, self.normal[nb_img,yaxis,:,0])
        plt.plot(xaxis,self.pred[nb_img,yaxis,:,0],color='green')

        plt.subplot(nb_row,nb_col,2,xlabel='Lung Abnormal')
        plt.plot(xaxis,self.abnormal[nb_img,yaxis,:,0])
        plt.plot(xaxis,self.pred[nb_img,yaxis,:,0],color='green')

        plt.subplot(nb_row,nb_col,3,xlabel='Lung Predict')
        plt.plot(xaxis,self.pred[nb_img,yaxis,:,0])

        plt.subplot(nb_row,nb_col,4,xlabel='Abnormal')
        plt.plot(xaxis,self.ab_pred[nb_img,yaxis,:,0])
        plt.show()

    def view_plot_3(self,nb_img,yaxis):
        plt.style.use("ggplot")
        plt.figure(figsize=(100,10))
        nb_row = 1
        nb_col = 3
        
        size = self.pred.shape[2]
        xaxis = np.arange(0,size)
        plt.subplot(nb_row,nb_col,1,xlabel='Lung Normal-Pred')
        plt.plot(xaxis, self.pred_nor[nb_img,yaxis,:,0])

        plt.subplot(nb_row,nb_col,2,xlabel='Lung Abnormal - Normal')
        plt.plot(xaxis,self.ab_nor[nb_img,yaxis,:,0])

        plt.subplot(nb_row,nb_col,3,xlabel='Lung Abnormal - Predict')
        plt.plot(xaxis,self.ab_pred[nb_img,yaxis,:,0])
        plt.show()

    def view_result(self,nb_img,yaxis):
        self.view_lung(nb_img)
        self.view_plot(nb_img,yaxis)
        self.view_plot_3(nb_img,yaxis)
        self.view_compare_normal(nb_img)
        self.view_compare_abnormal(nb_img)