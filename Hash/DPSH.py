import torch 
import torch.nn as nn 
import time, sys, os
os.environ["CUDA_VISIBLE_DEVICES"]='1'
    
sys.path.append("../")
from utils.backbone import *   
from utils.dataloader import * 
from utils.tools import *   
# torch.backends.cudnn.benchmark = True 
    
num_classes = {
    "cifar10-0": 10,
    "cifar10-1": 10,     
    "cifar10-2": 10,
    "nus-wide": 21,
    "flickr25k": 38,
    "imagenet100": 100 
}
    
def get_config():   
    config = {
        "model_name": "AlexNet",
        "dataset": "imagenet100",
        "save_path": "../save/DPSH",
        "code_save": "../code/DPSH",

        "eta": 0.1,
        "epoch": 100,  
        "batch_size": 256,
        "bits": 32,
        "step": 25,
        "lr": 1e-5,
        "topk": 1000
    }
    return config 
  
class DPSH(object):
    def __init__(self, num_train, train_labels):
        super(DPSH, self).__init__()

        self.bits = config["bits"]
        self.num_class = num_classes[config["dataset"]]
        self.epochs = config["epoch"]
        self.eta = config["eta"]
        self.U = torch.zeros(num_train, self.bits).cuda()
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

    def DPSHLoss(self, H, y, idx):
        self.U[idx, :] = H.data

        omiga = 1/2 * H.mm(self.U.t())
        s = (y.mm(self.Y.t()) > 0).float()

        loss1 = (torch.log(1 + torch.exp(omiga)) - s * omiga).mean()
        loss2 = (torch.sign(H) - H).pow(2).mean()
        return loss1 + config["eta"] * loss2 
    
    def train(self, trainloader, testloader, dbloader):
        # optimizer = torch.optim.SGD(self.model.parameters(), lr = 0.05, weight_decay = 1e-5) #1e-2 lr=0.05
        optimizer = torch.optim.RMSprop(self.model.parameters(), lr = config["lr"], weight_decay=1e-5) #1e-5
        best_map = 0 
        best_topk = 0 
        self.model.train()
        for epoch in range(self.epochs):
            current_time = time.strftime("%H:%M:%S", time.localtime(time.time()))
            print("{:d}/{:d}, DPSH, time:{}, bit:{:d}, dataset:{}, backbone:{}".format(epoch, self.epochs, current_time, self.bits, config["dataset"], config["model_name"]))
            
            train_loss = 0
            for idx, (img, label, index) in enumerate(trainloader):
                img = img.cuda()
                label = label.cuda()
                _, output = self.model(img) 
                loss = self.DPSHLoss(output, label, index)
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
        qB, qlabels = compute_result(testloader, self.model)
        # if not os.path.exists(os.path.join(config["code_save"], "{}_{}_{}_code.npy".format(config["model_name"], config["dataset"], config["bits"]))):
        dB, dlabels = compute_result(dbloader, self.model)
        np.save(os.path.join(config["code_save"], "{}_{}_{}_code.npy".format(config["model_name"], config["dataset"], config["bits"])), dB.numpy())
        np.save(os.path.join(config["code_save"], "{}_{}_{}_label.npy".format(config["model_name"], config["dataset"], config["bits"])), dlabels.numpy())
        # else:
        #     print("------")
        #     dB = np.load(os.path.join(config["code_save"], "{}_{}_{}_code.npy".format(config["model_name"], config["dataset"], config["bits"])))
        #     dlabels = np.load(os.path.join(config["code_save"], "{}_{}_{}_label.npy".format(config["model_name"], config["dataset"], config["bits"])))
        #     dB = torch.from_numpy(dB)
        #     dlabels = torch.from_numpy(dlabels)

        map = CalcMap(qB, dB, qlabels, dlabels)
        map_topk = CalcTopMap(qB, dB, qlabels, dlabels, 1000)
        print("test mAP:{:.4f}".format(map))
        print("test topk mAP:{:.4f}".format(map_topk))

    def save_model(self):
        model_path = os.path.join(config["save_path"], "{}_{}_{}_model.pth".format(config["model_name"], config["dataset"], config["bits"]))
        torch.save(self.model.state_dict(), model_path)

    def load_model(self):
        model_path = os.path.join(config["save_path"], "{}_{}_{}_model.pth".format(config["model_name"], config["dataset"], config["bits"]))
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path))

if __name__ == "__main__":
    # set_seed(100)
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
        test_dataset = Imagenet(os.path.join(data_path, config["dataset"], "test10.txt"),
                                    os.path.join(img_dir, config["dataset"]),
                                    transform)
        database_dataset = Imagenet(os.path.join(data_path, config["dataset"], "database.txt"),
                                os.path.join(img_dir, config["dataset"]),
                                transform)
        label_onehot = train_dataset.get_label()
        print(label_onehot.size())
   
    elif config["dataset"] == "cifar-10":
        transform = transforms.Compose([
                transforms.Resize(scale_size),
                transforms.CenterCrop(crop_size),
                transforms.ToTensor(),
            ]) 
        train_filename = ["data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4", "data_batch_5"]
        test_filename = ["test_batch"]
        train_dataset, test_dataset, database_dataset = cifar_dataset(config, "", train_filename, test_filename, transform, transform)                           
        label_onehot = train_dataset.get_label()

    num_train, num_test, num_db = len(train_dataset), len(test_dataset), len(database_dataset)
    trainloader = data.DataLoader(train_dataset, batch_size = config["batch_size"], shuffle = True, num_workers = 8)
    testloader = data.DataLoader(test_dataset, batch_size = config["batch_size"], shuffle = False, num_workers = 8)
    dbloader = data.DataLoader(database_dataset, batch_size = config["batch_size"], shuffle = False, num_workers = 8)

    hash_model = DPSH(num_train, label_onehot)
    hash_model.train(trainloader, testloader, dbloader)
    # hash_model.load_model()
    # hash_model.test(testloader, dbloader)