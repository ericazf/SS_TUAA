import torch 
import torch.nn as nn 
import time 
import sys 
import os 
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
    "flickr25k": 38,
}
    
def get_config():
    config = {
        "model_name": "VGG11",
        "dataset": "nus-wide",
        "save_path": "../save/DSDH",
        "code_save": "../code/DSDH",
        "bits": 64,

        "eta": 1e-2,
        "mu": 1e-2,
        "nu": 1,

        "epochs": 100,
        "batch_size": 24,
        "step": 25,
        "lr": 1e-5,
        "topk": 5000
    }
    return config 
   
class DSDH(object):
    def __init__(self, num_train, train_labels):
        super(DSDH, self).__init__()
        self.num_train = num_train
        self.bits = config["bits"]
        self.num_class = num_classes[config["dataset"]]
        self.epochs = config["epochs"]

        self.U = torch.zeros(num_train, self.bits).cuda()
        self.B = torch.randn(num_train, self.bits).sign().cuda()
        self.Y = train_labels.cuda()

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

    def loss(self, H, y, idx):
        self.U[idx, :] = H.data

        omiga = H.mm(self.U.t()) * 0.5
        s = (y.mm(self.Y.t()) > 0).float()
        
        #prevent exp overflow
        # omiga = torch.clamp(omiga, min = -100, max = 50)
        # loss1 = (torch.log(1 + torch.exp(omiga)) - s * omiga).mean()
        loss1 = (torch.log(1 + torch.exp(- omiga.abs())) + omiga.clamp(min = 0) - s * omiga).mean()
        loss2 = (self.B[idx, :] - H).pow(2).mean()
        loss = loss1 + config["eta"] * loss2
        return loss 

    def updateBandW(self):
        B = self.B 
        for iter in range(10):
            W = torch.inverse(B.t().mm(B) + config["nu"]/config["mu"] * torch.eye(self.bits).cuda()).mm(B.t()).mm(self.Y)

            for i in range(B.size(1)):
                P = W.mm(self.Y.t()) + config["eta"] / config["mu"] * self.U.t()
                p = P[i, :]
                w = W[i, :]
                W_prime = torch.cat((W[:i, :], W[i + 1:, :]), dim = 0)
                B_prime = torch.cat((B[:, :i], B[:, i + 1:]), dim = 1)
                B[:, i] = (p - B_prime @ W_prime @ w).sign()
        self.B = B
        self.W = W 

    def train(self, trainloader, testloader, dbloader):
        # optimizer = torch.optim.SGD(self.model.parameters(), lr = config["lr"], momentum = 0.9) #1e-2
        optimizer = torch.optim.RMSprop(self.model.parameters(), lr = config["lr"], weight_decay=1e-5) #1e-5
        
        best_map = 0
        best_topk = 0
        for epoch in range(self.epochs):
            current_time = time.strftime("%H:%M:%S", time.localtime(time.time()))
            print("{:d}/{:d}, DSDH, time:{}, bit:{:d}, dataset:{}, backbone:{}".format(epoch, self.epochs, current_time, self.bits, config["dataset"], config["model_name"]))
            
            self.model.train()
            train_loss = 0
            for idx, (img, label, index) in enumerate(trainloader):
                img = img.cuda()
                label = label.cuda()
                H, output = self.model(img) 
                loss = self.loss(output, label, index)
                train_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            self.updateBandW()
            
            train_loss = train_loss / len(trainloader)
            print("loss:{:.3f}".format(train_loss))
            if (epoch + 1)%config["step"] == 0:
                best_map, best_topk = validate(config, best_map, best_topk, testloader, dbloader, self.model)
                print("best_map:{:.3f}, best_topk:{:.3f}".format(best_map, best_topk))

    def test(self, testloader, dbloader):
        self.load_model()
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
        topk = CalcTopMap(qB, dB, qlabels, dlabels, config["topk"])
        print("test mAP:{:.4f}, topk_map:{:.4f}".format(map, topk))

    def save_model(self):
        model_path = os.path.join(config["save_path"], "{}_{}_{}_model.pth".format(config["model_name"], config["dataset"], config["bits"]))
        torch.save(self.model.state_dict(), model_path)

    def load_model(self):
        model_path = os.path.join(config["save_path"], "{}_{}_{}_model.pth".format(config["model_name"], config["dataset"], config["bits"]))
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path))

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

    elif config["dataset"] == "imagenet100":
        transform = transforms.Compose([
                transforms.Resize(scale_size),
                transforms.CenterCrop(crop_size),
                transforms.ToTensor(),
            ])
        train_dataset = Imagenet(os.path.join(data_path, config["dataset"], "train.txt"),
                                    os.path.join(img_dir, config["dataset"]),
                                    transform)
        test_dataset = Imagenet(os.path.join(data_path, config["dataset"], "test.txt"),
                                    os.path.join(img_dir, config["dataset"]),
                                    transform)
        database_dataset = Imagenet(os.path.join(data_path, config["dataset"], "database.txt"),
                                os.path.join(img_dir, config["dataset"]),
                                transform)
        label_onehot = train_dataset.get_label()
        print(label_onehot.size())

    num_train, num_test, num_db = len(train_dataset), len(test_dataset), len(database_dataset)
    trainloader = data.DataLoader(train_dataset, batch_size = config["batch_size"], shuffle = True, num_workers = 4)
    testloader = data.DataLoader(test_dataset, batch_size = config["batch_size"], shuffle = False, num_workers = 4)
    dbloader = data.DataLoader(database_dataset, batch_size = config["batch_size"], shuffle = False, num_workers = 4)

    hash_model = DSDH(num_train, label_onehot)
    hash_model.train(trainloader, testloader, dbloader)
    # hash_model.test(testloader, dbloader)
    # hash_model.finetune(trainloader, testloader, dbloader)