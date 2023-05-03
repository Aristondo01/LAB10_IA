import os
import cv2
import random
from sklearn.model_selection import train_test_split
import numpy as np
import torch
from torch.utils.data import TensorDataset

class ImageReader(object):
    def __init__(self):
        self.categories = ['Dog', 'Cat']
        self.img_width = 56
        self.img_height = 56
    
    def read_images(self,limit):
        self.images = []
        for category in self.categories:
            path = os.path.join(category)
            class_num = self.categories.index(category)
            for img in os.listdir(path):
                
                try:
                    img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                    new_array = cv2.resize(img_array, (self.img_width, self.img_height))
                    self.images.append([new_array, class_num])
                    limit -= 1
                except:
                    os.remove(os.path.join(path, img))
                    
                    
                    pass
                if limit == 0:
                    break
    
    def get_train_and_test(self):
        train_data, test_data = train_test_split(self.images, test_size = 0.2, random_state = 1234)
        X_train = []
        y_train = []
        for element in train_data:
            X_train.append(element[0])
            y_train.append(element[1])
        
        X_train, y_train = np.array(X_train), np.array(y_train)
        X_train = torch.LongTensor(X_train)/255.0
        y_train = torch.LongTensor(y_train)
        train_tensor = TensorDataset(X_train, y_train)

        X_test = []
        y_test = []
        for element in test_data:
            X_test.append(element[0])
            y_test.append(element[1]) 
        
        X_test, y_test = np.array(X_test), np.array(y_test)
        X_test = torch.LongTensor(X_test)/255.0
        y_test = torch.LongTensor(y_test)
        test_tensor = TensorDataset(X_test, y_test)

        return train_tensor, test_tensor