import time 
import torch 
import torch.nn as nn 
import copy 
import sys, os 
import numpy as np 

from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms 
import torch.utils.data as data
import torch.nn.functional as F 

os.environ["CUDA_VISIBLE_DEVICES"]='2'
sys.path.append("../")
from utils.backbone import *    
from utils.dataloader import * 
from utils.tools import * 
   
num_classes = {
    "cifar10-0": 10,
    "cifar10-1": 10,
    "cifar10-2": 10,
    "nus-wide": 21,
    "flickr25k": 38
}
        
def get_config():   
    config = {
        "model_name": "VGG11",
        "dataset": "flickr25k",
        "bits": 32,
        "save_path": "../save/ADSH",
        "code_save": "../code/ADSH", 
     
        "gamma": 200,
        "Tin": 3,
        "Tout":50,
        "num_sample": 2000,
        "batch_size": 24,
        "step": 10,
        "lr": 1e-4,
        "topk": 5000
    }
    return config 

def sample_dataloader(config, db_dataset, num_sample):
    db_data = db_dataset.data
    db_label = db_dataset.label
    sample_index = torch.randperm(np.shape(db_data)[0])[:num_sample]
    train_data = db_data[sample_index]
    train_label = db_label[sample_index]

    if config["dataset"].startswith("cifar10"):
        train_label = F.one_hot(torch.LongTensor(train_label), num_classes = 10).float()
    else:
        train_label = torch.tensor(train_label).float()

    class MyDataset(Dataset):
        def __init__(self, train_data, train_label):
            self.data = train_data
            self.label_onehot = train_label 

            self.transform = transforms.Compose([
                transforms.Resize(scale_size),
                transforms.CenterCrop(crop_size),
                transforms.ToTensor(),
            ])
        
        def __getitem__(self, index):
            img = self.data[index]
            if config["dataset"].startswith("cifar10"):
                img = Image.fromarray(img.astype("uint8")).convert("RGB")
            else:
                img = Image.open(img).convert("RGB")
            img = self.transform(img)
            label = self.label_onehot[index]
            return img, label, index 
        
        def __len__(self):
            return len(self.data)
        
        def get_label(self):
            return self.label_onehot

    train_dataset = MyDataset(train_data, train_label)
    trainloader = data.DataLoader(train_dataset, batch_size = config["batch_size"], shuffle = True, num_workers = 4)
    return trainloader, sample_index 

class ADSH(object):
    def __init__(self, num_db, db_labels):
        super(ADSH, self).__init__()
        self.bits = config["bits"] 
        self.num_sample = config["num_sample"]
        self.num_class = num_classes[config["dataset"]]
        self.Tin = config["Tin"]
        self.Tout = config["Tout"]
        self.num_db = num_db 

        self.U = torch.zeros(self.num_sample, self.bits).cuda()
        self.V = torch.randn(self.num_db, self.bits).cuda()
        self.Y = db_labels.cuda()

        if config["model_name"].startswith("VGG"):
            self.model = VGG(config["model_name"], self.bits).cuda()
        elif config["model_name"].startswith("AlexNet"):
            self.model = AlexNet(self.bits).cuda()
        elif config["model_name"].startswith("ResNet"):
            self.model = ResNet(config["model_name"], self.bits).cuda()
        elif config["model_name"].startswith("DenseNet"):
            self.model = DenseNet(config["model_name"], self.bits).cuda()
        elif config["model_name"].startswith("Incv3"):
            self.model = InceptionV3(config["model_name"], self.bits).cuda()
        
    def loss(self, H, y, idx, S):
        loss1 = (H.mm(self.V.t()) - self.bits * S).pow(2).sum()
        loss2 = (self.V[idx, :] - H).pow(2).sum()
        return (loss1 + config["gamma"] * loss2)/(S.size(0) * S.size(1))
    
    def updateV(self, expand_U, S):
        Q = - 2 * self.bits * S.t().mm(self.U) - 2 * config["gamma"] * expand_U
        # Q = self.bits * S.t().mm(self.U) + config["gamma"] * expand_U

        for i in range(self.bits):
            Vk = torch.cat((self.V[:, :i], self.V[:, i+1:]), dim = 1)
            Uk = torch.cat((self.U[:, :i], self.U[:, i+1:]), dim = 1)
            u = self.U[:, i]
            q = Q[:, i]
            v =  - (q.t() + 2 * Vk @ Uk.t() @ u.t()).sign()
            # v = (q.t() - Vk @ Uk.t() @ u.t()).sign()
            self.V[:,i] = v  

    def train(self, db_dataset, testloader):
        # optimizer = torch.optim.SGD(self.model.parameters(), lr = config["lr"]) SGD优化器效果不够好
        optimizer = torch.optim.Adam(self.model.parameters(), lr=config["lr"], weight_decay=1e-5)
        best_map = 0
        best_topk = 0 
        for tout in range(self.Tout):
            self.model.train()
            trainloader, sample_index = sample_dataloader(config, db_dataset, self.num_sample)
            train_label = trainloader.dataset.get_label().cuda()
            S = 1.0 * (train_label.mm(self.Y.t()) > 0)
            S = 2 * S - 1

            r = S.sum() / (1 - S).sum()
            S = S * (1 + r) - r 
            for epoch in range(self.Tin):
                train_loss = 0 
                for iter, (img, label, idx) in enumerate(trainloader):
                    img = img.cuda()
                    label = label.cuda()
                    H, output = self.model(img)
                    self.U[idx, :] = output.data

                    loss = self.loss(output, label, sample_index[idx], S[idx, :])
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()

                print("tout:{}, epoch:{}, loss:{:.3f}".format(tout, epoch, train_loss/len(trainloader)))
                
            expand_U = torch.zeros(self.V.size()).cuda()
            expand_U[sample_index, :] = self.U 
            self.updateV(expand_U, S)
                
            if (tout + 1) % config["step"] == 0:
                qB, query_labels = compute_result(testloader, self.model)
                map = CalcMap(qB, self.V.cpu(), query_labels, self.Y.cpu())
                topk = CalcTopMap(qB, self.V.cpu(), query_labels, self.Y.cpu(), config["topk"])
                if map > best_map:
                    best_map = map 
                    best_topk = topk
                    torch.save(self.model.state_dict(), os.path.join(config["save_path"], "{}_{}_{}_model.pth".format(config["model_name"], config["dataset"], config["bits"])))
                    np.save(os.path.join(config["code_save"], "{}_{}_{}_code.npy".format(config["model_name"], config["dataset"], config["bits"])), self.V.cpu().numpy())
                    np.save(os.path.join(config["code_save"], "{}_{}_{}_label.npy".format(config["model_name"], config["dataset"], config["bits"])), self.Y.cpu().numpy())
                print("best_map:{:.3f}, best_topk:{:.3f}".format(best_map, best_topk))

if __name__ == "__main__":
    set_seed(100) 
    config = get_config()

    data_path = "../data"
    img_dir = ""

    if config["model_name"] == "Incv3":
        scale_size = 300
        crop_size = 299
    else:
        scale_size = 256
        crop_size = 224

    if config["dataset"] == "nus-wide" or config["dataset"] == "flickr25k":
        transform = transforms.Compose(
        [
            transforms.Resize(scale_size),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
        ])   
        train_dataset = MultiDatasetLabel(os.path.join(data_path, config["dataset"], "train_img.txt"),
                                    os.path.join(data_path, config["dataset"], "train_label.txt"),
                                    os.path.join(img_dir, config["dataset"]),
                                    transform)
        test_dataset = MultiDatasetLabel(os.path.join(data_path, config["dataset"], "test_img.txt"),
                                    os.path.join(data_path, config["dataset"], "test_label.txt"),
                                    os.path.join(img_dir, config["dataset"]),
                                    transform)
        database_dataset = MultiDatasetLabel(os.path.join(data_path, config["dataset"], "database_img.txt"),
                                os.path.join(data_path, config["dataset"], "database_label.txt"),
                                os.path.join(img_dir, config["dataset"]),
                                transform)
        db_label_onehot = database_dataset.get_label()
        print(db_label_onehot.size())

    num_train, num_test, num_db = len(train_dataset), len(test_dataset), len(database_dataset)
    trainloader = data.DataLoader(train_dataset, batch_size = config["batch_size"], shuffle = True, num_workers = 4)
    testloader = data.DataLoader(test_dataset, batch_size = config["batch_size"], shuffle = False, num_workers = 4)
    dbloader = data.DataLoader(database_dataset, batch_size = config["batch_size"], shuffle = False, num_workers = 4)
    
    hash_model = ADSH(num_db, db_label_onehot)
    hash_model.train(database_dataset, testloader)