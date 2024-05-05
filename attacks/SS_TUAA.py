import os   
import torch 
import numpy as np
import torch.nn as nn 
import torch.nn.functional as F 
from torchvision import transforms 
import torch.utils.data as data 

import logging 
import torch.optim as optim 

# from attacks.Anchor import * 
from CAE import * 
from utils.backbone import * 
from utils.tools import *
from utils.dataloader import * 

os.environ["CUDA_VISIBLE_DEVICES"]='1'
 
num_train = {
    "flickr25k": 5000,
    "nus-wide": 10500,
} 
def get_config():  
    config = {
        "dataset": "flickr25k",
        "backbone": "VGG11",
        "bits":16,
        "save_path": "../save/DPSH",
        "code_save": "../code/DPSH",
     
        "anchor": "CAE",
        "anchor_path": "../anchorcode/DPSH/CAE",
        "log_path": "../log/DPSH/CAE",
        "UAP_path": "../results/DPSH/CAE",


        "T": 1,
        "loss": "F",  
        "epochs":10,
        "alpha": 1,
        "layer_name":"model.features.13",
        "single": False, 

        "batch_size": 24,
        "lr": 0.01,
        "eps": 10/255.0,
    }
    return config 

class HookTool:
    def __init__(self):
        self.fea = None 
    def hook_fun(self, module, fea_in, fea_out):
        self.fea = fea_out 

def get_feas_by_hook(model):
    fea_hooks = []
    for n, m  in model.named_modules():
        if isinstance(m, torch.nn.Conv2d):
        # or isinstance(m, torch.nn.Linear):
            if config["single"] and n != config["layer_name"]:
                continue
            cur_hook = HookTool()
            m.register_forward_hook(cur_hook.hook_fun)
            fea_hooks.append(cur_hook)
    print(len(fea_hooks))
    return fea_hooks

def get_feature(fea_hooks):         
    features = []
    for ht in fea_hooks:
        features.append(ht.fea)
    return features 

def fea_loss(adv_features, anchor_features):
    total_loss = 0
    for adv, anc in zip(adv_features, anchor_features):
        cos_loss = 1 - nn.CosineSimilarity(dim = 1)(adv.view(adv.size(0), -1), anc.view(anc.size(0), -1)).mean()
        total_loss = total_loss + cos_loss
    return total_loss/len(adv_features)


"""
Stability $\leftarrow$  PNI of anchor code & Fcl loss function

$\downarrow$

TUAA performance $\leftarrow$ Semantic-preserving ability of anchor code 
"""

class TUAA_UAP(object):
    def __init__(self, target_labels):
        super(TUAA_UAP, self).__init__()
        self.target_labels = target_labels 
        print("--------------", self.target_labels.size())

        self.class_dict = {"flickr25k":38, "nus-wide":21, "imagenet100":100}
        self.num_classes = self.class_dict[config["dataset"]]
        self.bits = config["bits"]
        self.lr = config["lr"]
        self.eps = config["eps"]

        self._build_model()
        self._load_data()
        self.fea_hooks = get_feas_by_hook(self.hashing_model)

    def _build_model(self):
        self.prototype = PrototypeNet(self.bits, self.num_classes).cuda()

        if config["backbone"].startswith("VGG"):
            self.hashing_model = VGG(config["backbone"], self.bits).cuda()
        elif config["backbone"].startswith("AlexNet"):
            self.hashing_model = AlexNet(self.bits).cuda()
        elif config["backbone"].startswith("ResNet"):
            self.hashing_model = ResNet(config["backbone"], self.bits).cuda()
        elif config["backbone"].startswith("DenseNet"):
            self.hashing_model = DenseNet(config["model_name"], self.bits).cuda()

        self.hashing_model.load_state_dict(torch.load(os.path.join(config["save_path"], "{}_{}_{}_model.pth".format(config["backbone"], config["dataset"], config["bits"]))))
        self.hashing_model.eval()

    def _load_data(self):    
        self.db_codes = np.load(os.path.join(config["code_save"], "{}_{}_{}_code.npy".format(config["backbone"], config["dataset"], config["bits"])))
        self.db_labels = np.load(os.path.join(config["code_save"], "{}_{}_{}_label.npy".format(config["backbone"], config["dataset"], config["bits"])))
        self.db_codes = torch.from_numpy(self.db_codes).float()
        self.db_labels = torch.from_numpy(self.db_labels).float()
        self.num_db = self.db_codes.size(0)

    def load_anchor(self):
        if config["anchor"] == "PNet":
            self.prototype = torch.load(os.path.join(config["anchor_path"], "prototypenet_{}_{}_{}.pth".format(config["backbone"], config["dataset"], config["bits"])))
            self.prototype.eval()

        elif config["anchor"] == "CAE":
            anchor_code = np.load(os.path.join(config["anchor_path"], "AnchorCode_{}_{}_{}_{}.npy".format(config["backbone"], config["dataset"], config["bits"], config["T"])))
            target_labels = np.load(os.path.join(config["anchor_path"], "TargetLabel_{}_{}_{}_{}.npy".format(config["backbone"], config["dataset"], config["bits"], config["T"])))
            self.label2idx = {}
            for num, label in enumerate(target_labels):
                label_str = self.label2str(label)
                self.label2idx[label_str] = num 
            self.anchor_record = torch.from_numpy(anchor_code)

        elif config["anchor"] == "CHCM":
            anchor_code = np.load(os.path.join(config["anchor_path"], "AnchorCode_{}_{}_{}.npy".format(config["backbone"], config["dataset"], config["bits"])))
            target_labels = np.load(os.path.join(config["anchor_path"], "TargetLabel_{}_{}_{}.npy".format(config["backbone"], config["dataset"], config["bits"])))
            self.label2idx = {}
            for num, label in enumerate(target_labels):
                label_str = self.label2str(label)
                self.label2idx[label_str] = num 
            self.anchor_record = torch.from_numpy(anchor_code)

    def label2str(self, label):
        label_str = ""
        for i in label:
            label_str = label_str + str(int(i))
        return label_str

    def get_anchor(self, target_labels):
        if config["anchor"] == "PNet":
            with torch.no_grad():
                _, batch_codes, _ = self.prototype(target_labels.cuda())
            batch_codes = batch_codes.detach().cpu().sign()

        else:
            batch_size = target_labels.size(0)
            batch_codes = torch.zeros(batch_size, self.bits)
            for i in range(batch_size):
                label = target_labels[i]
                label_str = self.label2str(label)
                batch_codes[i] = self.anchor_record[self.label2idx[label_str]]
        return batch_codes
    
    def CalcSim(self, label, target_label):
        sim = 1.0 * (label.mm(target_label.t()) > 0)
        return sim  

    def generate_code(self, trainloader, num_train):
        train_codes = torch.zeros(num_train, self.bit)
        train_labels = torch.zeros(num_train, self.num_classes)
        for iter, (img, label, index) in enumerate(trainloader):
            img = img.cuda()
            _, code = self.hashing_model(img)
            train_codes[index, :] = torch.sign(code.cpu().data)
            train_labels[index, :] = label 
        return train_codes, train_labels 
    
    def Loss(self, anchor, code):
        loss = (-torch.mean(anchor * code, dim = 1) + 1).sum() / code.size(0)
        return loss 

    def L_hag(self, anchor, code):
        mask = torch.sign(torch.abs(anchor - code) - 0.3)
        mask = (mask + 1)/2
        code = mask * code
        m = torch.sum(mask, dim = 1)
        loss = (- torch.sum(anchor * code, dim = 1)/(m + 1e-8) + 1).sum() / code.size(0)
        return loss

    '''
    Our CAE method and FCL loss function can be combined with two different implementations of Hamming distance loss function to improve the TUAA performance, i.e., 
                    1) Hamming distance loss with mask (HAG, AdvHash)
                    2) Hamming distance loss without mask (DHTA, ProS-GAN) 
    1) config["alpha"] = 1
    We adopt Hamming distance loss with mask in our paper since it tends to make the TUAP fall into the feature subspace of anchor code i.e., H(\delta) = h_a. 
    The benefit of this is that the FCL can directly affect the stability of TUAP.
    In contrast, the FCL will indirectly improve the stability of TUAP by improving the Hamming distance with h_\delta.
    2) config["alpha"] = 0.5
    The benefit of the second loss function is that experiments show its superior TUAA performance.
    '''

    def main(self, trainloader, num_train, testloader, num_test):
        self.load_anchor()
        UAP_records = torch.zeros(self.target_labels.size(0), 3, 224, 224)
        avg_record = [0, 0, 0, 0, 0, 0, 0]

        for num, target_label in enumerate(self.target_labels):
            print("====class: {}/{}".format(num, self.target_labels.size(0)))
            target_label = target_label.unsqueeze(0)
            anchor = self.get_anchor(target_label)

            perturbation = torch.zeros(1,3,224,224).cuda()
            perturbation.requires_grad = True
            optimizer = optim.Adam([perturbation], lr = 0.01)
            #------------------------------------------------------------
            start = time.time()
            for epoch in range(config["epochs"]):
                print("epoch:{}/{}".format(epoch, config["epochs"]))
                
                for i, (img, label, index) in enumerate(trainloader):
                    img = img.cuda()
                    adv_img = img + perturbation
                    
                    # with torch.no_grad():
                    _, _ = self.hashing_model(perturbation)
                    anchor_feature = get_feature(self.fea_hooks)

                    adv_img = torch.clamp(adv_img, min = 0, max = 1)
                    H, code = self.hashing_model(adv_img)
                    adv_feature = get_feature(self.fea_hooks)

                    # hloss = self.Loss(anchor.cuda(), code)
                    hloss = self.L_hag(anchor.cuda(), code)
                    floss = fea_loss(adv_feature, anchor_feature)

                    if config["loss"] == "F":
                        loss = hloss + config["alpha"] * floss 
                    elif config["loss"] == "H":
                        loss = hloss
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    perturbation.data = torch.clamp(perturbation.data, -self.eps, self.eps)
                    if i % 100 == 0:
                        hamm = self.calc_hamm(anchor, code.detach().cpu().sign()).mean()
                        print("iter:{}, hloss:{:.5f}, floss:{:.5f}, dis:{:.5f}".format(i, hloss, floss, hamm))
            
            end = time.time()
            UAP_records[num, :] = perturbation.detach().cpu()
            anchor_map_test, ori_map_test, tuap_map_test, HD_anchor, PNI_anchor, HD_p, PNI_p = self.test(testloader, num_test, anchor, perturbation, target_label, "Test")

            avg_record[0], avg_record[1], avg_record[2], avg_record[3], avg_record[4], avg_record[5], avg_record[6] = \
            avg_record[0]+anchor_map_test, avg_record[1]+ori_map_test, avg_record[2]+tuap_map_test, avg_record[3]+HD_anchor, avg_record[4]+PNI_anchor, avg_record[5]+HD_p, avg_record[6]+PNI_p 
            logger.info("%d \t %d \t %.1f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f", num, rand_idx, end-start, anchor_map_test, ori_map_test, tuap_map_test, HD_anchor, PNI_anchor, HD_p, PNI_p)
        
        logger.info("Total \t %d \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f", rand_idx, avg_record[0]/self.target_labels.size(0), avg_record[1]/self.target_labels.size(0), avg_record[2]/self.target_labels.size(0), avg_record[3]/self.target_labels.size(0), avg_record[4]/self.target_labels.size(0), avg_record[5]/self.target_labels.size(0), avg_record[6]/self.target_labels.size(0))
        if not os.path.exists(config["UAP_path"]):
            os.mkdir(config["UAP_path"])
        np.save(os.path.join(config["UAP_path"], "UAP_{}_{}_{}.npy".format(config["backbone"], config["dataset"], config["bits"])), UAP_records)


    def calc_hamm(self, ori_code, adv_code):
        K = ori_code.size(1)
        hamm = (K - torch.sum(ori_code * adv_code, dim = 1))/2
        return hamm

    def test(self, testloader, num_test, anchor, perturbation, target_label, mode = "Test"):
        qB = torch.zeros(num_test, self.bits)
        oB = torch.zeros(num_test, self.bits)
        target_labels = torch.zeros(num_test, self.num_classes)
        
        HD_anchor = 0
        HD_p = 0

        with torch.no_grad():
            _, code_per = self.hashing_model(perturbation)
            code_per = torch.sign(code_per.cpu().data)
            
            for i, (img, label, index) in enumerate(testloader):
                img = img.cuda()
                adv_img = img + perturbation
                adv_img = torch.clamp(adv_img, min = 0, max = 1)

                H, code_adv = self.hashing_model(adv_img)
                H, code_ori = self.hashing_model(img)
                code_adv = torch.sign(code_adv.cpu().data)
                code_ori = torch.sign(code_ori.cpu().data)

                qB[index, :] = code_adv 
                oB[index, :] = code_ori
                target_labels[index, :] = target_label
                
                HD_anchor = HD_anchor + self.calc_hamm(anchor, code_adv).sum()
                HD_p = HD_p + self.calc_hamm(code_per, code_adv).sum()

        PNI_anchor = (anchor.mm(self.db_codes.t())/self.bits == 1).float()
        PNI_anchor = torch.sum(PNI_anchor, dim = 1).mean()
        PNI_p = (code_per.mm(self.db_codes.t())/self.bits == 1).float()
        PNI_p = torch.sum(PNI_p, dim = 1).mean()
        
        if mode == "Test":
            anchor_map = CalcMap(anchor, self.db_codes, target_label, self.db_labels)
            ori_map = CalcMap(oB, self.db_codes, target_labels, self.db_labels)
            t_map = CalcMap(qB, self.db_codes, target_labels, self.db_labels)
            return anchor_map, ori_map, t_map, HD_anchor/num_test, PNI_anchor, HD_p/num_test, PNI_p
        

    def test_save(self):
        self.load_anchor()
        avg_record = [0, 0, 0, 0, 0, 0, 0]
        perturbations = np.load(os.path.join(config["UAP_path"], "UAP_{}_{}_{}.npy".format(config["backbone"], config["dataset"], config["bits"])))
        for num, target_label in enumerate(self.target_labels):
            target_label = target_label.unsqueeze(0)
            anchor = self.get_anchor(target_label)
            per = perturbations[num]
            per = torch.from_numpy(per).unsqueeze(0).cuda()
            anchor_map_test, ori_map_test, tuap_map_test, HD_anchor, PNI_anchor, HD_p, PNI_p = self.test(testloader, num_test, anchor, per, target_label, "Test")

            logger.info("%d \t %d \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f", num, rand_idx, anchor_map_test, ori_map_test, tuap_map_test, HD_anchor, PNI_anchor, HD_p, PNI_p)
            avg_record[0], avg_record[1], avg_record[2], avg_record[3], avg_record[4], avg_record[5], avg_record[6] =\
            avg_record[0]+anchor_map_test, avg_record[1]+ori_map_test, avg_record[2]+tuap_map_test, avg_record[3]+HD_anchor, avg_record[4]+PNI_anchor, avg_record[5]+HD_p, avg_record[6]+PNI_p 
        
        logger.info("Total \t %d \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f", rand_idx, avg_record[0]/self.target_labels.size(0), avg_record[1]/self.target_labels.size(0), avg_record[2]/self.target_labels.size(0), avg_record[3]/self.target_labels.size(0), avg_record[4]/self.target_labels.size(0),
                    avg_record[5]/self.target_labels.size(0), avg_record[6]/self.target_labels.size(0))
    

if __name__ == "__main__":
    rand_idx = torch.randint(low = 0, high = 1000, size = (1,))
    set_seed(100) 
    config = get_config()

    data_path = "../data"
    img_dir = "/data1/zhufei/datasets"

    if config["dataset"] == "nus-wide" or config["dataset"] == "flickr25k":
        transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
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
    trainloader = data.DataLoader(train_dataset, batch_size = config["batch_size"], shuffle = True, num_workers=4)
    testloader = data.DataLoader(test_dataset, batch_size = config["batch_size"], shuffle = False, num_workers = 4)
    dbloader = data.DataLoader(database_dataset, batch_size = config["batch_size"], shuffle = False, num_workers = 4)

    if not os.path.exists(config["log_path"]):
            os.mkdir(config["log_path"])
    logger = logging.getLogger(__name__)
    logging.basicConfig(level = logging.INFO, datefmt = "%Y/%m/%d %H:%M:%S", 
                        format = "%(asctime)s - %(message)s",
                        handlers = [
                            logging.FileHandler(os.path.join(config["log_path"], "{}_{}_{}.log".format(config["backbone"], config["dataset"], config["bits"]))),
                            logging.StreamHandler()
                        ])
    current_time = time.strftime("%H:%M:%S", time.localtime())
    logger.info("Current time:%s \t idx:%d \t anchor_map \t ori_map \t test_map \t dis_test \t dis_train", current_time, rand_idx)
    logger.info(config.values())

    if config["dataset"] == "flickr25k":
        target_labels = np.loadtxt("../target_label/flickr25k/label_38.txt")
        target_labels = torch.from_numpy(target_labels).to(torch.float32)
    elif config["dataset"] == "nus-wide":
        target_labels = np.loadtxt("../target_label/nus-wide/label_21.txt")
        target_labels = torch.from_numpy(target_labels).to(torch.float32)
    

    uap = TUAA_UAP(target_labels)
    uap.main(trainloader, num_train, testloader, num_test)
    # uap.test_save()
