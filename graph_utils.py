import numpy as np
import scipy.sparse as sp
from scipy.sparse import *
import torch
from math import ceil
from Params import args
from torch.utils import data
import pickle

def data_process(test):

    size = test.shape[0]
    # 创建长度为31882的列表
    list_of_lists = [None] * size
    # 遍历稀疏矩阵 A 的非零元素
    for row, col, value in zip(test.row, test.col, test.data):
        if value != 0 :  # 检查值是否为非零
            list_of_lists[row] = col  # 将值添加到对应列索引的子列表中
    return list_of_lists
def get_use(behaviors_data):

    behavior_mats = {}
        
    behaviors_data = (behaviors_data != 0) * 1

    behavior_mats['A'] = matrix_to_tensor(normalize_adj(behaviors_data))
    behavior_mats['AT'] = matrix_to_tensor(normalize_adj(behaviors_data.T))
    behavior_mats['A_ori'] = None

    return behavior_mats
def get_composite(behaviors,behaviors_mats,A):
    A_sum = torch.sum(torch.stack([behaviors_mats[i]['A'] for i in range(len(behaviors))]), dim=0)
    for i in range(len(behaviors)):
        A[i]={}
        A[i]['A'] = A_sum
        A[i]['AT'] = A_sum.T
        A[i]['A_ori'] = None
    return A

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    rowsum_diag = sp.diags(np.power(rowsum+1e-8, -0.5).flatten())

    colsum = np.array(adj.sum(0))
    colsum_diag = sp.diags(np.power(colsum+1e-8, -0.5).flatten())


    return rowsum_diag.dot(adj).dot(colsum_diag)


def matrix_to_tensor(cur_matrix):
    if type(cur_matrix) != sp.coo_matrix:
        cur_matrix = cur_matrix.tocoo()  
    indices = torch.from_numpy(np.vstack((cur_matrix.row, cur_matrix.col)).astype(np.int64))  
    values = torch.from_numpy(cur_matrix.data)  
    shape = torch.Size(cur_matrix.shape)

    return torch.sparse_coo_tensor(indices, values, shape).to(torch.float32).cuda(device=args.device)

class RecDataset(data.Dataset):
    def __init__(self, data, num_item, train_mat=None, num_ng=1, is_training=True):
        super(RecDataset, self).__init__()

        self.data = np.array(data)
        self.num_item = num_item
        self.train_mat = train_mat
        self.is_training = is_training

    def ng_sample(self):
        assert self.is_training, 'no need to sampling when testing'
        dok_trainMat = self.train_mat.todok()
        length = self.data.shape[0]
        self.neg_data = np.random.randint(low=0, high=self.num_item, size=length)

        for i in range(length):  #
            uid = self.data[i][0]
            iid = self.neg_data[i]
            if (uid, iid) in dok_trainMat:
                while (uid, iid) in dok_trainMat:
                    iid = np.random.randint(low=0, high=self.num_item)
                    self.neg_data[i] = iid
                self.neg_data[i] = iid

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        user = self.data[idx][0]
        item_i = self.data[idx][1]

        if self.is_training:
            neg_data = self.neg_data
            item_j = neg_data[idx]
            return user, item_i, item_j
        else:
            return user, item_i

    def getMatrix(self):
        pass

    def getAdj(self):
        pass

    def sampleLargeGraph(self):

        def makeMask():
            pass

        def updateBdgt():
            pass

        def sample():
            pass

    def constructData(self):
        pass


class RecDataset_beh(data.Dataset):
    # beh=behaviors,data=train_data,num_item=item_num,behaviors_data=behaviors_data
    def __init__(self, beh, data, num_item, behaviors_data=None, num_ng=1, is_training=True):
        super(RecDataset_beh, self).__init__()
        self.data = np.array(data)
        self.num_item = num_item
        self.is_training = is_training
        self.beh = beh
        self.behaviors_data = behaviors_data
        self.length = self.data.shape[0]
        self.neg_data = [None] * self.length
        self.pos_data = [None] * self.length

    def ng_sample(self):
        assert self.is_training, 'no need to sampling when testing'
#对每一个有购买行为的用户都设置4个正样本和负样本
        for i in range(self.length):
            self.neg_data[i] = [None] * len(self.beh)
            self.pos_data[i] = [None] * len(self.beh)

        for index in range(len(self.beh)):

            train_u, train_v = self.behaviors_data[index].nonzero()
            beh_dok = self.behaviors_data[index].todok()

            set_pos = np.array(list(set(train_v)))
            # 选择length个负样本和正样本，其中把item的id去重之后随机选择length个为正样本，然后从所有的item中选择length个作为负样本，选length长度的原因是因为要与购买行为长度对齐。
            self.pos_data_index = np.random.choice(set_pos, size=self.length, replace=True, p=None)
            self.neg_data_index = np.random.randint(low=0, high=self.num_item, size=self.length)

            for i in range(self.length):  #
                # uid 为购买行为矩阵下面的所有userid
                uid = self.data[i][0]
                # 对于每一个有购买行为的user，获取一个index行为下的负样本和正样本，其中正样本一定为在该行为下存在与某个user有交互行为的item id。
                iid_neg = self.neg_data[i][index] = self.neg_data_index[i]
                iid_pos = self.pos_data[i][index] = self.pos_data_index[i]
                # 对于这个负采样物品，如果与购买行为下的user存在其他行为的交互，则再重新选择一个。
                if (uid, iid_neg) in beh_dok:
                    while (uid, iid_neg) in beh_dok:
                        iid_neg = np.random.randint(low=0, high=self.num_item)
                        self.neg_data[i][index] = iid_neg
                    self.neg_data[i][index] = iid_neg
                # 对于购买行为，与user交互的那个item就是正样本。  而对于其他行为，如果该正样本与user没有交互，则从与该user有交互的节点中选一个。如果该节点与任何item都没有交互，则赋值为-1.
                if index == (len(self.beh) - 1):
                    self.pos_data[i][index] = train_v[i]
                elif (uid, iid_pos) not in beh_dok:
                    if len(self.behaviors_data[index][uid].data) == 0:
                        self.pos_data[i][index] = -1
                    else:
                        t_array = self.behaviors_data[index][uid].toarray()
                        pos_index = np.where(t_array != 0)[1]
                        iid_pos = np.random.choice(pos_index, size=1, replace=True, p=None)[0]
                        self.pos_data[i][index] = iid_pos

    # 这样之后可以得到对每一个有购买行为的user，都有4个正样本和4个负样本。其中正样本为该user在其他行为下的交互。负样本为该user在其他行为下的不存在交互的节点。
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        user = self.data[idx][0]
        item_i = self.pos_data[idx]

        if self.is_training:
            item_j = self.neg_data[idx]
            return user, item_i, item_j
        else:
            return user, item_i
