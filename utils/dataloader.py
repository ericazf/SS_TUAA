import os 
import torch
import torchvision
import pickle
import numpy as np
from torch.utils import data
from torchvision import transforms
from PIL import Image 
import torch.nn.functional as F 
from torchvision.datasets.cifar import CIFAR10

class CIFAR10Dataset(data.Dataset):
    def __init__(self, data_dir, filename, transform):
        super(CIFAR10Dataset, self).__init__()
        self.transform = transform 

        self.file_list = []
        for file in filename:
            data_path = os.path.join(data_dir, file)
            with open(data_path, "rb") as f:
                filedict = pickle.load(f, encoding = "bytes")
                self.file_list.append(filedict)
        
        self.data = np.array([])
        self.label = np.array([])
        for batch in self.file_list:
            if len(self.data) == 0:
                self.data = np.array(batch[b"data"])
                self.label = np.array(batch[b"labels"])
            else:
                self.data = np.append(self.data, batch[b"data"], axis = 0)
                self.label = np.append(self.label, batch[b"labels"], axis = 0)
        self.data = np.reshape(self.data, (-1, 3, 32, 32))
        self.data = np.transpose(self.data, (0,2,3,1))
    
    def __getitem__(self, index):
        image = self.data[index]
        image = Image.fromarray(image.astype("uint8")).convert("RGB")
        image = self.transform(image)

        label = self.label[index]
        label_onehot = F.one_hot(torch.LongTensor([label]), num_classes = 10).float()
        label_onehot = label_onehot[0]
        return image, label_onehot, index 
    
    def __len__(self):
        return len(self.data)

    def get_label(self):
        self.label_onehot = F.one_hot(torch.LongTensor(self.label), num_classes = 10).float()
        return self.label_onehot

def cifar_dataset(config, data_dir, train_filename, test_filename, train_transform, test_transform):
    if config["dataset"] == "cifar10-0":
        # test:10000, train: 5000, database:50000
        train_size = 500
        test_size = 1000
    elif config["dataset"] == "cifar10-1":
        # test:1000, train:5000, database:59000
        train_size = 500
        test_size = 100
    elif config["dataset"] == "cifar10-2":
        # test:10000 train:50000
        train_size = 5000
        test_size = 1000

    train_dataset = CIFAR10Dataset(data_dir, train_filename, train_transform)
    test_dataset = CIFAR10Dataset(data_dir, test_filename, test_transform)
    database_dataset = CIFAR10Dataset(data_dir, train_filename, test_transform)
    
    dbdata = np.concatenate((train_dataset.data, test_dataset.data))
    dblabel = np.concatenate((train_dataset.label, test_dataset.label))

    first = True 
    for label in range(10):
        index = np.where(label == dblabel)[0]
        N = index.shape[0]
        perm = np.random.permutation(N)
        index = index[perm]

        if first:
            test_index = index[:test_size]
            train_index = index[test_size: train_size + test_size]
            database_index = index[test_size:]
        else:
            test_index = np.concatenate((test_index, index[:test_size]))
            train_index = np.concatenate((train_index, index[test_size: train_size + test_size]))
            database_index = np.concatenate((database_index, index[test_size:]))
        first = False 
    
    train_dataset.data = dbdata[train_index]
    test_dataset.data = dbdata[test_index]
    database_dataset.data = dbdata[database_index]
    train_dataset.label = dblabel[train_index]
    test_dataset.label = dblabel[test_index]
    database_dataset.label = dblabel[database_index]
    
    return train_dataset, test_dataset, database_dataset

class TrainMulti(data.Dataset):
    def __init__(self, image, label, transform):
        super(TrainMulti, self).__init__()
        self.data = image
        self.label = label
        self.transform = transform
        
    def __getitem__(self, index):
        img_path = self.data[index]
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)

        label = self.label[index]
        label = torch.FloatTensor(label)
        return img, label, index 

    def __len__(self):
        return len(self.data)

    def get_label(self):
        return torch.tensor(self.label).float()

class MultiDatasetLabel(data.Dataset):
    def __init__(self, image_path, label_path, data_dir, transform):
        super(MultiDatasetLabel, self).__init__()
        self.data_dir = data_dir 
        self.image_path = image_path 
        self.label_path = label_path 
        fp = open(self.image_path, "r")
        self.data = [os.path.join(self.data_dir, x.strip()) for x in fp]
        fp.close()

        fp = open(self.label_path, "r")
        self.label = [x.strip().split(" ") for x in fp]
        self.label = [[int(i) for i in x] for x in self.label]
        fp.close()

        self.data = np.array(self.data)
        self.label = np.array(self.label)
        self.transform = transform
        
    def __getitem__(self, index):
        img_path = self.data[index]
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)

        label = self.label[index]
        label = torch.FloatTensor(label)
        return img, label, index 

    def __len__(self):
        return len(self.data)

    def get_label(self):
        return torch.tensor(self.label).float()
    
class Imagenet(data.Dataset):
    def __init__(self, image_path, data_dir, transform):
        super(Imagenet, self).__init__()
        self.data_dir = data_dir 
        self.image_path = image_path
        with open(self.image_path, "r") as f:
            img_label = f.readlines() 
        self.data = [os.path.join(self.data_dir, img.split(" ")[0]) for img in img_label]
        self.label = [[int(i) for i in label.split(" ")[1:]] for label in img_label]
        
        self.label = np.array(self.label)
        self.data = np.array(self.data)
        self.transform = transform 

    def __getitem__(self,index):
        img_path = self.data[index]
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)

        label = self.label[index]
        label = torch.FloatTensor(label)
        return img, label, index        
    
    def __len__(self):
        return len(self.data)
    
    def get_label(self):
        return torch.tensor(self.label).float() 

class TrainImagenet(data.Dataset):
    def __init__(self, image_label, data_dir, transform):
        super(TrainImagenet, self).__init__()
        self.data_dir = data_dir 
        img_label = image_label 

        self.data = [os.path.join(self.data_dir, img.split(" ")[0]) for img in img_label]
        self.label = [[int(i) for i in label.split(" ")[1:]] for label in img_label]
        
        self.label = np.array(self.label)
        self.data = np.array(self.data)
        self.transform = transform 

    def __getitem__(self,index):
        img_path = self.data[index]
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)

        label = self.label[index]
        label = torch.FloatTensor(label)
        return img, label, index        
    
    def __len__(self):
        return len(self.data)
    
    def get_label(self):
        return torch.tensor(self.label).float() 
