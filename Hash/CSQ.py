import time 
import torch 
import random 
import torch.nn as nn 
import sys 
import os 

os.environ["CUDA_VISIBLE_DEVICES"]='0'
sys.path.append("../")
from scipy.linalg import hadamard 
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
        "model_name":"DenseNet161",
        "dataset": "flickr25k",
        "save_path": "../save/CSQ",
        "lambda": 0.0001,

        "epochs": 100,
        "batch_size": 24,
        "bits": 32,
        "step": 25,
        "lr": 1e-5,
        "topk": 5000
    }
    return config 

class CSQ(object):
    def __init__(self):
        super(CSQ, self).__init__()
        self.num_class = num_classes[config["dataset"]]
        self.bits = config["bits"]
        self.is_single_label = config["dataset"] not in {"nus-wide", "flickr25k"}
        self.epochs = config["epochs"]

        self.hash_targets = self.get_hash_targets(self.num_class, self.bits).cuda()
        self.multi_label_random_center = torch.randint(2, (self.bits,)).float().cuda()
        self.BCE = torch.nn.BCELoss()

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

    def CSQLoss(self, H, y, idx):
        hash_center = self.label2center(y)
        center_loss = self.BCE(0.5 * (H + 1), 0.5 * (hash_center + 1))
        Q_loss = (H.abs() - 1).pow(2).mean()
        return center_loss + config["lambda"] * Q_loss 

    def get_hash_targets(self, num_class, bits):
        H_K = hadamard(bits)
        H_2K = np.concatenate((H_K, -H_K), 0)
        hash_targets = torch.from_numpy(H_2K[:num_class]).float()
        
        if H_2K.shape[0] < num_class:
            print("-------")
            hash_targets.resize_(num_class, bits)
            for k in range(20):
                for index in range(H_2K.shape[0], num_class):
                    ones = torch.ones(bits)
                    sa = random.sample(list(range(bits)), bits//2)
                    ones[sa] = -1 
                    hash_targets[index] = ones 
                c = []
                for i in range(num_class):
                    for j in range(num_class):
                        if i < j:
                            TF = sum(hash_targets[i] != hash_targets[j])
                            c.append(TF)
                c = np.array(c)

                if c.min() > bits /4 and c.mean() >= bits/2:
                    print(c.min(), c.mean())
                    break 
        return hash_targets 
    
    def label2center(self, y):
        if self.is_single_label:
            hash_center = self.hash_targets[torch.argmax(y, dim = 1)]
        else:
            center_sum = y.mm(self.hash_targets)
            random_center = self.multi_label_random_center.repeat(center_sum.size(0), 1)
            print(random_center)
            center_sum[center_sum == 0] = random_center[center_sum == 0]
            hash_center = 2 * (center_sum > 0).float() - 1

        return hash_center 
    
    def train(self, trainloader, testloader, dbloader):
        # optimizer = torch.optim.SGD(self.model.parameters(), lr = config["lr"])
        optimizer = torch.optim.RMSprop(self.model.parameters(), lr = config["lr"], weight_decay=1e-5) #1e-5
        best_map = 0
        best_topk = 0 
        for epoch in range(self.epochs):
            current_time = time.strftime("%H:%M:%S", time.localtime(time.time()))
            print("{:d}/{:d}, CSQ, time:{}, bit:{:d}, dataset:{}, backbone:{}".format(epoch, self.epochs, current_time, self.bits, config["dataset"], config["model_name"]))
            
            self.model.train()
            train_loss = 0
            for idx, (img, label, index) in enumerate(trainloader):
                img = img.cuda()
                label = label.cuda()
                H, output = self.model(img) 
                loss = self.CSQLoss(output, label, index)
                train_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            train_loss = train_loss / len(trainloader)
            print("loss:{:.3f}".format(train_loss))
            if (epoch + 1)%config["step"] == 0:
                best_map, best_topk = validate(config, best_map, best_topk, testloader, dbloader, self.model)
                print("best_map:{:.3f}, best_topk:{:.3f}".format(best_map, best_topk))
    
    def load_model(self):
        model_path = os.path.join(config["save_path"], "{}_{}_{}_model.pth".format(config["model_name"], config["dataset"], config["bits"]))
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path))

    def test(self, testloader, dbloader):
        self.load_model()
        qB, qlabels = compute_result(testloader, self.model)
        dB, dlabels = compute_result(dbloader, self.model)
        map = CalcMap(qB, dB, qlabels, dlabels)
        topk = CalcTopMap(qB, dB, qlabels, dlabels, config["topk"])
        print("test mAP:{:.4f}, topk_map:{:.4f}".format(map, topk))

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
        label_onehot = train_dataset.get_label()
        print(label_onehot.size())


    num_train, num_test, num_db = len(train_dataset), len(test_dataset), len(database_dataset)
    trainloader = data.DataLoader(train_dataset, batch_size = config["batch_size"], shuffle = True, num_workers = 4)
    testloader = data.DataLoader(test_dataset, batch_size = config["batch_size"], shuffle = False, num_workers = 4)
    dbloader = data.DataLoader(database_dataset, batch_size = config["batch_size"], shuffle = False, num_workers = 4)
   
    hash_model = CSQ()
    hash_model.train(trainloader, testloader, dbloader)
    # hash_model.load_model()
    # hash_model.test(testloader, dbloader)