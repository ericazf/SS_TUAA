import time 
import sys, os 
import torch
import torch.nn as nn
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
    "flickr25k":38
}

def get_config():
    config = {
        "model_name": "DenseNet161",
        "dataset": "flickr25k",
        "save_path": "../save/HashNet",
        "code_save": "../code/HashNet",
   
        "epochs": 100,
        "batch_size": 24,
        "bits": 32,
        "step": 25,
        "lr": 1e-5,
        "topk": 5000
    }
    return config 

class HashNet(object):
    def __init__(self, num_train, train_labels):
        super(HashNet, self).__init__()
        self.bits = config["bits"]
        self.epochs = config["epochs"]
        
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


    def HashNetLoss(self, H, y, idx):
        S = (y.mm(y.t()) > 0).float()
        omiga = H.mm(H.t())

        sim_pos = (S == 1)
        dissim_pos = (S == 0)
        w1 = S.numel()/sim_pos.sum().float()
        w0 = S.numel()/dissim_pos.sum().float()
        W = torch.zeros(S.size()).cuda()
        W[sim_pos] = w1 
        W[dissim_pos] = w0 
        
        loss = (W * (torch.log(1 + torch.exp(- omiga.abs())) + omiga.clamp(min = 0) - S * omiga)).mean()
        return loss 
    
    def train(self, trainloader, testloader, dbloader):
        # optimizer = torch.optim.SGD(self.model.parameters(), lr = config["lr"]) #1e-2
        optimizer = torch.optim.RMSprop(self.model.parameters(), lr = config["lr"], weight_decay=1e-5) #1e-5
        best_map = 0
        best_topk = 0
        for epoch in range(self.epochs):
            current_time = time.strftime("%H:%M:%S", time.localtime(time.time()))
            print("{:d}/{:d}, HashNet, time:{}, bit:{:d}, dataset:{}".format(epoch, self.epochs, current_time, self.bits, config["dataset"]))
            
            self.model.train()
            train_loss = 0
            for idx, (img, label, index) in enumerate(trainloader):
                img = img.cuda()
                label = label.cuda()
                H, output = self.model(img) 
                loss = self.HashNetLoss(output, label, index)
                train_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            train_loss = train_loss / len(trainloader)
            print("loss:{:.3f}".format(train_loss))
            if (epoch + 1)%config["step"] == 0:
                best_map, best_topk = validate(config, best_map, best_topk, testloader, dbloader, self.model)
                print("best_map:{:.3f}, best_topk:{:.3f}".format(best_map, best_topk))

    def test(self, testloader, dbloader):
        self.load_model()
        # qB, qlabels = compute_result(testloader, self.model)
        # dB, dlabels = compute_result(dbloader, self.model)
        qB, qlabels = compute_result(testloader, self.model)
        if not os.path.exists(os.path.join(config["code_save"], "{}_{}_{}_code.npy".format(config["model_name"], config["dataset"], config["bits"]))):
            dB, dlabels = compute_result(dbloader, self.model)
            np.save(os.path.join(config["code_save"], "{}_{}_{}_code.npy".format(config["model_name"], config["dataset"], config["bits"])), dB.numpy())
            np.save(os.path.join(config["code_save"], "{}_{}_{}_label.npy".format(config["model_name"], config["dataset"], config["bits"])), dlabels.numpy())
        else:
            print("------")
            dB = np.load(os.path.join(config["code_save"], "{}_{}_{}_code.npy".format(config["model_name"], config["dataset"], config["bits"])))
            dlabels = np.load(os.path.join(config["code_save"], "{}_{}_{}_label.npy".format(config["model_name"], config["dataset"], config["bits"])))
            dB = torch.from_numpy(dB)
            dlabels = torch.from_numpy(dlabels)
        map = CalcMap(qB, dB, qlabels, dlabels)
        print("test mAP:{:.4f}".format(map))

    def save_model(self):
        model_path = os.path.join(config["save_path"], "{}_{}_{}_model.pth".format(config["model_name"], config["dataset"], config["bits"]))
        torch.save(self.model.state_dict(), model_path)

    def load_model(self):
        model_path = os.path.join(config["save_path"], "{}_{}_{}_model.pth".format(config["model_name"], config["dataset"], config["bits"]))
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path))

if __name__ == "__main__":
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

    hash_model = HashNet(num_train, label_onehot)
    hash_model.train(trainloader, testloader, dbloader)
    