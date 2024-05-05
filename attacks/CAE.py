import torch    
import torch.nn as nn 
import sys,os
import numpy as np
from torchvision import transforms
from torch.utils import data 
from torch.distributions import Categorical
from torch.autograd import Variable
os.environ["CUDA_VISIBLE_DEVICES"]='0'
     
sys.path.append("../")
from utils.tools import * 
from utils.dataloader import * 
from utils.backbone import * 
from utils.module import * 
      
num_classes = {
    "cifar10-0": 10,
    "cifar10-1": 10,
    "cifar10-2": 10,
    "nus-wide": 21,
    "flickr25k":38,
    "imagenet100": 100
}

num_train = {
    "flickr25k": 5000,  
    "nus-wide": 10500,
    "imagenet100": 13000
}
 
def get_config():
    config = {
        "dataset": "nus-wide",
        "bits": 16,
        "model_name":"ResNet18",
        "save_path": "../save/DPSH",

        "code_save": "../code/DPSH",
        "anchor_save": "../anchorcode/DPSH",

        "epochs":100,
        "steps":300,
        "batch_size":12,
        "lr": 1e-4,
        "T":1,

        "iteration": 7,
        "epsilon": 8/255.0,
        "alpha": 2/255.0,
        "num": 2,
        "threshold":0.3
    }
    return config 



#--------------------------prototype-----------------------------
def CalcSim(batch_label, train_label):
    S = (batch_label.mm(train_label.t()) > 0).float()
    return S

class Prototype(nn.Module):
    def __init__(self, dbloader, num_db):
        super(Prototype, self).__init__()
        self.bits = config["bits"]
        self.num_class = num_classes[config["dataset"]]
        
        self._build_model()
        self.dB = np.load(os.path.join(config["code_save"], "{}_{}_{}_code.npy".format(config["model_name"], config["dataset"], config["bits"])))
        self.db_labels = np.load(os.path.join(config["code_save"], "{}_{}_{}_label.npy".format(config["model_name"], config["dataset"], config["bits"])))
        self.dB = torch.from_numpy(self.dB)
        self.db_labels = torch.from_numpy(self.db_labels)

    def _build_model(self):
        self.prototype = PrototypeNet(self.bits, self.num_class).cuda()

        if config["model_name"].startswith("VGG"):
            self.hash_model = VGG(config["model_name"], self.bits).cuda()
        elif config["model_name"].startswith("AlexNet"):
            self.hash_model = AlexNet(self.bits).cuda()
        elif config["model_name"].startswith("ResNet"):
            self.hash_model = ResNet(config["model_name"], self.bits).cuda()
        elif config["model_name"].startswith("DenseNet"):
            self.hash_model = DenseNet(config["model_name"], self.bits).cuda()
        elif config["model_name"].startswith("Incv3"):
            self.hash_model = InceptionV3(config["model_name"], self.bits).cuda()
        self.hash_model.load_state_dict(torch.load(os.path.join(config["save_path"], "{}_{}_{}_model.pth".format(config["model_name"], config["dataset"], config["bits"]))))
        self.hash_model.eval()
        
    def hashloss(self, anchorcode, targetlabel, traincodes, trainlabels):
        S = (targetlabel.mm(trainlabels.t()) > 0).float()
        omiga = 1/2 * anchorcode.mm(traincodes.t())
        logloss = (torch.log(1 + torch.exp(-omiga.abs())) + omiga.clamp(min = 0) - S * omiga).sum()
        return logloss 

    def train_prototype(self, train_loader, num_train, target_labels):
        optimizer_l = torch.optim.Adam(self.prototype.parameters(), lr=config["lr"], betas=(0.5, 0.999))
        epochs = 100
        steps = 300
        batch_size = 64
        lr_steps = epochs * steps
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer_l, milestones=[lr_steps / 2, lr_steps * 3 / 4], gamma=0.1)
        criterion_l2 = torch.nn.MSELoss()

        B, train_labels = self.generate_code(train_loader, num_train)
        B = B.cuda()
        train_labels = train_labels.cuda()

        for epoch in range(epochs):
            for i in range(steps):
                select_index = np.random.choice(range(target_labels.size(0)), size=batch_size)
                batch_target_label = target_labels.index_select(0, torch.from_numpy(select_index)).cuda()

                optimizer_l.zero_grad()
                S = CalcSim(batch_target_label, train_labels)
                _, target_hash_l, label_pred = self.prototype(batch_target_label)

                logloss = self.hashloss(target_hash_l, batch_target_label, B, train_labels)/(num_train * batch_size)
                regterm = (torch.sign(target_hash_l) - target_hash_l).pow(2).sum() / (1e4 * batch_size)
                classifer_loss = criterion_l2(label_pred, batch_target_label)
                loss = logloss + classifer_loss + regterm

                loss.backward()
                optimizer_l.step()
                if i % 100 == 0:
                    print('epoch: {:2d}, step: {:3d}, lr: {:.5f}, logloss:{:.5f}, regterm: {:.5f}'
                        .format(epoch, i, scheduler.get_last_lr()[0], logloss, regterm))
                scheduler.step()
        self.save_prototype()
        
    def test_prototype(self, total_labels):
        self.prototype.eval()
        label_size = total_labels.size(0)
        qB = torch.zeros(label_size, self.bits)

        avg_tol = 0
        zero_sum = 0
        for i in range(total_labels.size(0)):
            label = total_labels[i].cuda()
            _, code, _ = self.prototype(label)
            code = code.cpu().sign().data
            qB[i, :] = code
            freq = (code.unsqueeze(0).mm(self.dB.t())/self.bits == 1).float()
            freq = torch.sum(freq, dim = 1)
            avg_tol = avg_tol + freq.sum()
            zero_sum += (freq == 0).float().sum()

        if config["dataset"] == "imagenet100":
            map = CalcTopMap(qB, self.dB, total_labels, self.db_labels, 1000)
        else:
            map = CalcMap(qB, self.dB, total_labels, self.db_labels)
        print("prototypenet test map:{:.4f}".format(map))
        print("avg_tol:{:.2f}, zero_sum:{:2f}".format(avg_tol/total_labels.size(0), zero_sum))

    def generate_code(self, trainloader, num_train):
        hashcode = torch.zeros(num_train, self.bits)
        hashlabels = torch.zeros(num_train, self.num_class)
        for iter, (img, label, idx) in enumerate(trainloader):
            img = img.cuda()
            label = label.cuda()
            H, code = self.hash_model(img)
            hashcode[idx, :] = code.cpu().data.sign()
            hashlabels[idx, :] = label.cpu().data
        return hashcode, hashlabels 
    
    def save_prototype(self):
        anchor_path = os.path.join(config["anchor_save"], "PNet")
        if not os.path.exists(anchor_path):
            os.mkdir(anchor_path)
        torch.save(self.prototype, os.path.join(anchor_path, "prototypenet_{}_{}_{}.pth".format(config["model_name"], config["dataset"], config["bits"])))

    def load_prototype(self):
        anchor_path = os.path.join(config["anchor_save"], "PNet")
        self.prototype = torch.load(os.path.join(anchor_path, "prototypenet_{}_{}_{}.pth".format(config["model_name"], config["dataset"], config["bits"])))


#-----------------voting--------------------
class CHCM(nn.Module):
    def __init__(self, dbloader, num_db):
        super(CHCM, self).__init__()
        self.bits = config["bits"]
        self.epochs = config["epochs"]
        self.steps = config["steps"]
        self.num_class = num_classes[config["dataset"]]
        
        self._build_model()
        # self.dB, self.db_labels = self.generate_code(dbloader, num_db)
        self.dB = np.load(os.path.join(config["code_save"], "{}_{}_{}_code.npy".format(config["model_name"], config["dataset"], config["bits"])))
        self.dB = torch.from_numpy(self.dB)
        self.db_labels = np.load(os.path.join(config["code_save"], "{}_{}_{}_label.npy".format(config["model_name"], config["dataset"], config["bits"])))
        self.db_labels = torch.from_numpy(self.db_labels)

    def _build_model(self):
        self.prototype = PrototypeNet(self.bits, self.num_class).cuda()

        if config["model_name"].startswith("VGG"):
            self.hash_model = VGG(config["model_name"], self.bits).cuda()
        elif config["model_name"].startswith("AlexNet"):
            self.hash_model = AlexNet(self.bits).cuda()
        elif config["model_name"].startswith("ResNet"):
            self.hash_model = ResNet(config["model_name"], self.bits).cuda()
        elif config["model_name"].startswith("DenseNet"):
            self.hash_model = DenseNet(config["model_name"], self.bits).cuda()
        elif config["model_name"].startswith("Incv3"):
            self.hash_model = InceptionV3(config["model_name"], self.bits).cuda()
        self.hash_model.load_state_dict(torch.load(os.path.join(config["save_path"], "{}_{}_{}_model.pth".format(config["model_name"], config["dataset"], config["bits"]))))
        self.hash_model.eval()

    def generate_code(self, trainloader, num_train):
        hashcode = torch.zeros(num_train, self.bits)
        hashlabels = torch.zeros(num_train, self.num_class)
        for iter, (img, label, idx) in enumerate(trainloader):
            img = img.cuda()
            label = label.cuda()
            H, code = self.hash_model(img)
            hashcode[idx, :] = code.cpu().data.sign()
            hashlabels[idx, :] = label.cpu().data
        return hashcode, hashlabels 
    
    def train(self, trainloader, num_train, target_labels):
        label_size = target_labels.size(0)
        aB = torch.zeros(label_size, self.bits)
        traincodes, trainlabels = self.generate_code(trainloader, num_train)

        avg_tol = 0
        zero_sum = 0
        for i in range(label_size):
            label = target_labels[i].unsqueeze(0)
            w = torch.sum(label * trainlabels, dim = 1)/torch.sum(torch.sign(label + trainlabels), dim = 1)
            w = w.unsqueeze(1)
            w1 = (w > 0).float()
            w2 = 1 - w1 
            # c1 = traincodes.size(0)/w1.sum()
            # c2 = traincodes.size(0)/w2.sum()   
            c1 = w2.sum()
            c2 = traincodes.size(0) - c1 
            anchor = torch.sign(torch.sum(c1*w1*traincodes - c2*w2*traincodes, dim = 0))
            aB[i] = anchor 

            freq = (anchor.unsqueeze(0).mm(self.dB.t())/self.bits == 1).float()
            freq = torch.sum(freq, dim = 1)
            avg_tol = avg_tol + freq.sum()
            zero_sum += (freq == 0).float().sum()

        if config["dataset"] == "imagenet100":
            map = CalcTopMap(aB, self.dB, target_labels, self.db_labels, 1000)
        else:
            map = CalcMap(aB, self.dB, target_labels, self.db_labels)
        print("CHCM anchor t-MAP:{:.4f}".format(map))
        print("avg_tol:{:.2f}, zero_sum:{:2f}".format(avg_tol/label_size, zero_sum))

        anchor_path = os.path.join(config["anchor_save"], "CHCM")
        if not os.path.exists(anchor_path):
            os.mkdir(anchor_path)
        np.save(os.path.join(anchor_path, "AnchorCode_{}_{}_{}.npy".format(config["model_name"], config["dataset"], config["bits"])), aB.numpy())
        np.save(os.path.join(anchor_path, "TargetLabel_{}_{}_{}.npy".format(config["model_name"], config["dataset"], config["bits"])), target_labels.numpy())
        return aB 
    
#----------------------Voting---------------------
class Voting(nn.Module):
    def __init__(self, dbloader, num_db):
        super(Voting, self).__init__()
        self.bits = config["bits"]
        self.epochs = config["epochs"]
        self.steps = config["steps"]
        self.num_class = num_classes[config["dataset"]]
        
        self._build_model()
        # self.dB, self.db_labels = self.generate_code(dbloader, num_db)
        self.dB = np.load(os.path.join(config["code_save"], "{}_{}_{}_code.npy".format(config["model_name"], config["dataset"], config["bits"])))
        self.dB = torch.from_numpy(self.dB)
        self.db_labels = np.load(os.path.join(config["code_save"], "{}_{}_{}_label.npy".format(config["model_name"], config["dataset"], config["bits"])))
        self.db_labels = torch.from_numpy(self.db_labels)

    def _build_model(self):
        self.prototype = PrototypeNet(self.bits, self.num_class).cuda()

        if config["model_name"].startswith("VGG"):
            self.hash_model = VGG(config["model_name"], self.bits).cuda()
        elif config["model_name"].startswith("AlexNet"):
            self.hash_model = AlexNet(self.bits).cuda()
        elif config["model_name"].startswith("ResNet"):
            self.hash_model = ResNet(config["model_name"], self.bits).cuda()
        elif config["model_name"].startswith("DenseNet"):
            self.hash_model = DenseNet(config["model_name"], self.bits).cuda()
        elif config["model_name"].startswith("Incv3"):
            self.hash_model = InceptionV3(config["model_name"], self.bits).cuda()
        self.hash_model.load_state_dict(torch.load(os.path.join(config["save_path"], "{}_{}_{}_model.pth".format(config["model_name"], config["dataset"], config["bits"]))))
        self.hash_model.eval()

    def generate_code(self, trainloader, num_train):
        hashcode = torch.zeros(num_train, self.bits)
        hashlabels = torch.zeros(num_train, self.num_class)
        for iter, (img, label, idx) in enumerate(trainloader):
            img = img.cuda()
            label = label.cuda()
            H, code = self.hash_model(img)
            hashcode[idx, :] = code.cpu().data.sign()
            hashlabels[idx, :] = label.cpu().data
        return hashcode, hashlabels 

    def CalcSim(self, label, target_label):
        sim = 1.0 * (label.mm(target_label.t()) > 0)
        return sim 

    def train(self, trainloader, num_train, target_labels):
        label_size = target_labels.size(0)
        aB = torch.zeros(label_size, self.bits)
        traincodes, trainlabels = self.generate_code(trainloader, num_train)

        avg_tol = 0
        zero_sum = 0
        for i in range(label_size):
            label = target_labels[i].unsqueeze(0)
            sim = label.mm(trainlabels.t()) > 0
            sim = sim.squeeze().unsqueeze(1) 
            anchor = torch.sign(torch.sum(sim*traincodes, dim = 0))
            aB[i] = anchor
        
            freq = (anchor.unsqueeze(0).mm(self.dB.t())/self.bits == 1).float()
            freq = torch.sum(freq, dim = 1)
            avg_tol = avg_tol + freq.sum()
            zero_sum += (freq == 0).float().sum()

        if config["dataset"] == "imagenet100":
            map = CalcTopMap(aB, self.dB, target_labels, self.db_labels, 1000)
        else:
            map = CalcMap(aB, self.dB, target_labels, self.db_labels)
        print("Voting anchor t-MAP:{:.4f}".format(map))
        print("avg_tol:{:.2f}, zero_sum:{:2f}".format(avg_tol/label_size, zero_sum))

        anchor_path = os.path.join(config["anchor_save"], "Voting")
        if not os.path.exists(anchor_path):
            os.mkdir(anchor_path)
        np.save(os.path.join(anchor_path, "AnchorCode_{}_{}_{}.npy".format(config["model_name"], config["dataset"], config["bits"])), aB.numpy())
        np.save(os.path.join(anchor_path, "TargetLabel_{}_{}_{}.npy".format(config["model_name"], config["dataset"], config["bits"])), target_labels.numpy())
        return aB 

class CAE(object):
    def __init__(self, dbloader, num_db):
        super(CAE, self).__init__()
        self.bits = config["bits"]
        self.epochs = config["epochs"]
        self.steps = config["steps"]
        self.num_class = num_classes[config["dataset"]]
        self.T = config["T"]
        
        self._build_model()
        # self.dB, self.db_labels = self.generate_code(dbloader, num_db)
        self.dB = np.load(os.path.join(config["code_save"], "{}_{}_{}_code.npy".format(config["model_name"], config["dataset"], config["bits"])))
        self.db_labels = np.load(os.path.join(config["code_save"], "{}_{}_{}_label.npy".format(config["model_name"], config["dataset"], config["bits"])))
        self.dB = torch.from_numpy(self.dB)
        self.db_labels = torch.from_numpy(self.db_labels)

    def _build_model(self):
        self.prototype = PrototypeNet(self.bits, self.num_class).cuda()

        if config["model_name"].startswith("VGG"):
            self.hash_model = VGG(config["model_name"], self.bits).cuda()
        elif config["model_name"].startswith("AlexNet"):
            self.hash_model = AlexNet(self.bits).cuda()
        elif config["model_name"].startswith("ResNet"):
            self.hash_model = ResNet(config["model_name"], self.bits).cuda()
        elif config["model_name"].startswith("DenseNet"):
            self.hash_model = DenseNet(config["model_name"], self.bits).cuda()
        elif config["model_name"].startswith("Incv3"):
            self.hash_model = InceptionV3(config["model_name"], self.bits).cuda()
        self.hash_model.load_state_dict(torch.load(os.path.join(config["save_path"], "{}_{}_{}_model.pth".format(config["model_name"], config["dataset"], config["bits"]))))
        self.hash_model.eval()

    def generate_code(self, trainloader, num_train):
        hashcode = torch.zeros(num_train, self.bits)
        hashlabels = torch.zeros(num_train, self.num_class)
        for iter, (img, label, idx) in enumerate(trainloader):
            img = img.cuda()
            label = label.cuda()
            H, code = self.hash_model(img)
            hashcode[idx, :] = code.cpu().data.sign()
            hashlabels[idx, :] = label.cpu().data
        return hashcode, hashlabels 

    def calc_codes(self):   
        dbcodes = self.dB 
        unique_codes = dbcodes.unique(dim = 0)
        sim = 1.0 * (unique_codes.mm(dbcodes.t())/self.bits == 1)
        sim = torch.sum(sim, dim = 1) 
        index = (1.0 * (sim >= self.T)).to(torch.bool) 
        unique_codes = unique_codes[index].to(torch.float32)
        print(unique_codes.size())

        index = (unique_codes.mm(dbcodes.t())/self.bits == 1).float()
        unique_labels = 1.0 * (index.mm(self.db_labels) > 0)
        print(unique_labels.size())
        return unique_codes, unique_labels
    
    def random_select(self):
        index = np.random.choice(range(self.dB.size(0)), size = 10000, replace = False)
        traincodes = self.dB[index]
        trainlabels = self.db_labels[index]
        return traincodes, trainlabels 

    def train(self, trainloader, num_train, target_labels):
        label_size = target_labels.size(0)
        aB = torch.zeros(label_size, self.bits)
        traincodes, trainlabels = self.generate_code(trainloader, num_train)
        unique_codes,unique_labels = self.calc_codes()

        avg_tol = 0
        zero_sum = 0
        for i in range(label_size):
            print("----------", i, label_size)
            label = target_labels[i].unsqueeze(0)
            index = label.mm(unique_labels.t()) > 0
            cand_codes = unique_codes[index.squeeze()]
           
            anchor = self.calc_anchor(cand_codes, label, traincodes, trainlabels)  
            aB[i] = anchor 

            freq = (anchor.unsqueeze(0).mm(self.dB.t())/self.bits == 1).float()
            freq = torch.sum(freq, dim = 1)
            avg_tol = avg_tol + freq.sum()
            zero_sum += (freq == 0).float().sum()
        
        if config["dataset"] == "imagenet100":
            map = CalcTopMap(aB, self.dB, target_labels, self.db_labels, 1000)
        else:
            map = CalcMap(aB, self.dB, target_labels, self.db_labels)
        print("CAE anchor t-MAP:{:.4f}".format(map))
        print("avg_tol:{:.2f}, zero_sum:{:2f}".format(avg_tol/label_size, zero_sum))

        anchor_path = os.path.join(config["anchor_save"], "CAE")
        if not os.path.exists(anchor_path):
            os.mkdir(anchor_path)
        np.save(os.path.join(anchor_path, "AnchorCode_{}_{}_{}_{}.npy".format(config["model_name"], config["dataset"], config["bits"], config["T"])), aB.numpy())
        np.save(os.path.join(anchor_path, "TargetLabel_{}_{}_{}_{}.npy".format(config["model_name"], config["dataset"], config["bits"], config["T"])), target_labels.numpy())
        return aB 
    
    def test(self):
        anchor_path = os.path.join(config["anchor_save"], "CAE")
        aB = np.load(os.path.join(anchor_path, "AnchorCode_{}_{}_{}_{}.npy".format(config["model_name"], config["dataset"], config["bits"], config["T"])))
        target_labels = np.load(os.path.join(anchor_path, "TargetLabel_{}_{}_{}_{}.npy".format(config["model_name"], config["dataset"], config["bits"], config["T"])))
        aB = torch.from_numpy(aB)
        target_labels = torch.from_numpy(target_labels)
        map = CalcMap(aB, self.dB, target_labels, self.db_labels)
        print("CAE anchor t-MAP:{:.4f}".format(map))

    def calc_anchor(self, unique_codes, target_class, codes, labels):
        target_class = target_class.repeat(unique_codes.size(0), 1)
        map_record = CalcAnchorMap(unique_codes, codes, target_class, labels)
        sort_index = torch.sort(-map_record).indices
        sort_map = map_record[sort_index]
        sort_codes = unique_codes[sort_index]
        return sort_codes[0]

if __name__ == "__main__":
    seed = torch.randint(low = 0, high = 1000, size = (1,))
    set_seed(100)
    config = get_config()

    data_path = "../data"
    img_dir = "/data1/zhufei/datasets"

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
        
        if not os.path.exists(os.path.join("../TUAP_train", config["dataset"], "train_img.txt")):
            fp = open(os.path.join(data_path, config["dataset"], "database_img.txt"), "r")
            img = [os.path.join(img_dir, config["dataset"], x.strip()) for x in fp]
            fp.close()
            fp = open(os.path.join(data_path, config["dataset"], "database_label.txt"), "r")
            label = [x.strip().split(" ") for x in fp]
            label = [[int(i) for i in x] for x in label]
            fp.close()  
            index = np.random.choice(range(len(img)), size = num_train[config["dataset"]], replace = False)
            train_img = np.array(img)[index]
            train_label = np.array(label)[index]
            np.savetxt(os.path.join("../TUAP_train", config["dataset"], "train_img.txt"), train_img, fmt = "%s")
            np.savetxt(os.path.join("../TUAP_train", config["dataset"], "train_label.txt"), train_label, fmt = "%d")
        else:
            train_img = np.loadtxt(os.path.join("../TUAP_train", config["dataset"], "train_img.txt"), dtype = str)
            train_label = np.loadtxt(os.path.join("../TUAP_train", config["dataset"], "train_label.txt"), dtype = float)   

        train_dataset = TrainMulti(train_img, train_label, transform)
        # train_dataset = MultiDatasetLabel(os.path.join("/data1/zhufei/datasets/data", config["dataset"], "train_img.txt"),
        #                             os.path.join("/data1/zhufei/datasets/data", config["dataset"], "train_label.txt"),
        #                             os.path.join("/data1/zhufei/datasets", config["dataset"]),
        #                             transform)
        test_dataset = MultiDatasetLabel(os.path.join(data_path, config["dataset"], "test_img.txt"),
                                    os.path.join(data_path, config["dataset"], "test_label.txt"),
                                    os.path.join(img_dir, config["dataset"]),
                                    transform)
        database_dataset = MultiDatasetLabel(os.path.join(data_path, config["dataset"], "database_img.txt"),
                                os.path.join(data_path, config["dataset"], "database_label.txt"),
                                os.path.join(img_dir, config["dataset"]),
                                transform)
        train_label = train_dataset.get_label()
        test_label = test_dataset.get_label()
        db_label = database_dataset.get_label() 
        unique_labels = db_label.unique(dim = 0)

    num_train, num_test, num_db = len(train_dataset), len(test_dataset), len(database_dataset)
    trainloader = data.DataLoader(train_dataset, batch_size = config["batch_size"], shuffle = True, num_workers = 4)
    testloader = data.DataLoader(test_dataset, batch_size = config["batch_size"], shuffle = False, num_workers = 4)
    dbloader = data.DataLoader(database_dataset, batch_size = config["batch_size"], shuffle = False, num_workers = 4)

    unique_labels = db_label.unique(dim = 0)
    if config["dataset"] == "flickr25k":
        unique_labels = unique_labels[1:]

    if not os.path.exists("../target_label"):
        target_idx = np.random.choice(range(unique_labels.size(0)), size = 10)
        target_labels = unique_labels[target_idx]
    else:
        if config["dataset"] == "flickr25k":
            target_labels = np.loadtxt("../target_label/flickr25k/label_38.txt")
            target_labels = torch.from_numpy(target_labels).to(torch.float32)
        elif config["dataset"] == "nus-wide":
            target_labels = np.loadtxt("../target_label/nus-wide/label_21.txt")
            target_labels = torch.from_numpy(target_labels).to(torch.float32)
        
    # proto = Prototype(dbloader, num_db)
    # proto.train_prototype(trainloader, num_train, target_labels)
    # proto.test_prototype(target_labels)

    cae = CAE(dbloader, num_db)
    cae.train(trainloader, num_train, target_labels)
    # cae.test()
    
    # chcm = CHCM(dbloader, num_db)
    # chcm.train(trainloader, num_train, target_labels)

    # voting = Voting(dbloader, num_db)
    # voting.train(trainloader, num_train, unique_labels)
