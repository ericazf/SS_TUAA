import torch 
import numpy as np 
import time 
import os 
import math 
import torch.nn.functional as F 
import random 

def CalcHammingDist(B1, B2):
    q = B2.size(1)
    distH = 0.5 * (q - B1.mm(B2.t()))
    return distH
       
def CalcMap(qB, dB, query_labels, db_labels):
    map = 0 
    num_query = qB.size(0)
    gnd = 1.0 * (query_labels.mm(db_labels.t()) > 0)
    hamm = CalcHammingDist(qB, dB)
    for i in range(num_query):
        gnd_i = gnd[i]
        tsum = torch.sum(gnd_i)
        count = torch.linspace(1, tsum, int(tsum))
        if tsum == 0:
            continue
        hamm_i = hamm[i]
        sort = torch.sort(hamm_i)
        gnd_i = gnd_i[sort.indices]
        # print(gnd_i[:10])
        t_index = torch.Tensor(torch.nonzero(gnd_i, as_tuple=False).float()) + 1
        t_index = t_index.squeeze()
        map_ = torch.mean(count/t_index)
        map = map + map_ 
    map = map/num_query 
    return map

def CalcAnchorMap(qB, dB, query_labels, db_labels):
    num_query = qB.size(0)
    gnd = 1.0 * (query_labels.mm(db_labels.t()) > 0) 
    hamm = CalcHammingDist(qB, dB)
    
    map_record = torch.zeros(qB.size(0))
    for i in range(num_query):
        gnd_i = gnd[i]
        tsum = torch.sum(gnd_i) # tsum=3 count = 1 2 3
        count = torch.linspace(1, tsum, int(tsum))
        if tsum == 0:
            continue 
        hamm_i = hamm[i]
        sort = torch.sort(hamm_i) # default ascend
        gnd_i = gnd_i[sort.indices]
        t_index = torch.Tensor(torch.nonzero(gnd_i, as_tuple = False).float()) + 1
        t_index = t_index.squeeze()
        map_ = torch.mean(count/t_index) # the dimension must be the same 
        map_record[i] = map_
    return map_record

def CalcTopMap(qB, dB, query_labels, db_labels, topk):
    map = 0 
    num_query = qB.size(0)
    gnd = 1.0 * (query_labels.mm(db_labels.t()) > 0)
    hamm = CalcHammingDist(qB, dB)
    for i in range(num_query):
        gnd_i = gnd[i]
        hamm_i = hamm[i]
        sort = torch.sort(hamm_i)

        gnd_i = gnd_i[sort.indices][:topk]
        tsum = torch.sum(gnd_i)
        count = torch.linspace(1, tsum, int(tsum))
        if tsum == 0:
            continue 
        t_index = torch.Tensor(torch.nonzero(gnd_i, as_tuple=False).float()) + 1
        t_index = t_index.squeeze()
        map_ = torch.mean(count/t_index)
        map = map + map_
    map = map / num_query 
    return map 

def CalcAnchorMapTop(qB, dB, query_labels, db_labels, topk):
    num_query = qB.size(0)
    gnd = 1.0 * (query_labels.mm(db_labels.t()) > 0)
    hamm = CalcHammingDist(qB, dB)

    map_record = torch.zeros(qB.size(0))
    for i in range(num_query):
        gnd_i = gnd[i]
        hamm_i = hamm[i]
        sort = torch.sort(hamm_i)

        gnd_i = gnd_i[sort.indices][:topk]
        tsum = torch.sum(gnd_i)
        count = torch.linspace(1, tsum, int(tsum))
        if tsum == 0:
            continue 
        t_index = torch.Tensor(torch.nonzero(gnd_i, as_tuple=False).float()) + 1
        t_index = t_index.squeeze()
        map_ = torch.mean(count/t_index)
        map_record[i] = map_
    return map_record 

def compute_result(dataloader, model):
    hashcodes = torch.zeros([])
    labels = torch.zeros([])
    model.eval()
    for idx, (img, label, index) in enumerate(dataloader):
        with torch.no_grad():
            img = img.cuda()
            _, output = model(img)
        if idx == 0:
            hashcodes = output.data.cpu().sign()
            labels = label.data.cpu()
        else:
            hashcodes = torch.cat((hashcodes, output.data.cpu().sign()), 0)
            labels = torch.cat((labels, label.cpu().data), 0) 
    return hashcodes, labels 

def validate(config, best_map, best_topk, testloader, dbloader, model):
    qB, query_labels = compute_result(testloader, model)
    dB, database_labels = compute_result(dbloader, model)
    topk = config['topk']

    map = CalcMap(qB, dB, query_labels, database_labels)
    top_map = CalcTopMap(qB, dB, query_labels, database_labels, topk)

    if map > best_map:
        best_map = map
        best_topk = top_map 
        save_path = config["save_path"]
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(model.state_dict(), os.path.join(save_path, "{}_{}_{}_model.pth".format(config["model_name"], config["dataset"], config["bits"])))
    return best_map, best_topk

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)  
    torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

