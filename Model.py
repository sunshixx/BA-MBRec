import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.autograd import Variable
from Params import args
from torch.nn.parameter import Parameter
import math
def to_var(x, requires_grad=True):
    if torch.cuda(device=args.device).is_available():
        x = x.cuda(device=args.device)
    return Variable(x, requires_grad=requires_grad)


class myModel(nn.Module):
    def __init__(self, userNum, itemNum, behavior, behavior_mats, A):
        super(myModel, self).__init__()
        self.A= A
        self.userNum = userNum
        self.itemNum = itemNum
        self.behavior = behavior
        self.behavior_mats = behavior_mats
        # self.embedding_dict = self.init_embedding()
        # self.weight_dict = self.init_weight()
        self.gcn = GCN(self.userNum, self.itemNum, self.behavior, self.behavior_mats, A)

    def forward(self):

        user_embed, item_embed, user_embeds, item_embeds = self.gcn()

        return user_embed, item_embed, user_embeds, item_embeds

    def para_dict_to_tenser(self, para_dict):

        tensors = []
        for beh in para_dict.keys():
            tensors.append(para_dict[beh])
        tensors = torch.stack(tensors, dim=0)

        return tensors.float()

    def update_params(self, lr_inner, first_order=False, source_params=None, detach=False):
        if source_params is not None:
            for tgt, src in zip(self.named_parameters(), source_params):
                name_t, param_t = tgt
                grad = src
                if first_order:
                    grad = to_var(grad.detach().data)
                tmp = param_t - lr_inner * grad
                self.set_param(self, name_t, tmp)
        else:

            for name, param in self.named_parameters()(self):
                if not detach:
                    grad = param.grad
                    if first_order:
                        grad = to_var(grad.detach().data)
                    tmp = param - lr_inner * grad
                    self.set_param(self, name, tmp)
                else:
                    param = param.detach_()
                    self.set_param(self, name, param)


class GCN(nn.Module):
    def __init__(self, userNum, itemNum, behavior, behavior_mats, A):
        super(GCN, self).__init__()
        self.userNum = userNum
        self.itemNum = itemNum
        self.hidden_dim = args.hidden_dim
        self.A = A
        self.behavior = behavior
        self.behavior_mats = behavior_mats
        self.user_embedding, self.item_embedding = self.init_embedding()
        self.i_concatenation_w, self.u_concatenation_w, = self.init_weight()
        self.sigmoid = torch.nn.Sigmoid()
        self.act = torch.nn.PReLU()
        self.dropout = torch.nn.Dropout(args.drop_rate)
        self.gnn_layer = eval(args.gnn_layer)

        self.layers = nn.ModuleList()
        self.layers.append(GCNLayer(args.hidden_dim, args.hidden_dim, self.userNum, self.itemNum, self.behavior,
                                        self.behavior_mats, matrix_typ='behavior'))
        for i in range(1, len(self.gnn_layer)):
            self.layers.append(GCNLayer(args.hidden_dim, args.hidden_dim, self.userNum, self.itemNum, self.behavior,
                                        self.A, matrix_typ='composite'))

    def init_embedding(self):
        user_embedding = torch.nn.Embedding(self.userNum, args.hidden_dim)
        item_embedding = torch.nn.Embedding(self.itemNum, args.hidden_dim)
        nn.init.xavier_uniform_(user_embedding.weight)
        nn.init.xavier_uniform_(item_embedding.weight)

        return user_embedding, item_embedding

    def init_weight(self):
        i_concatenation_w = nn.Parameter(torch.Tensor(len(eval(args.gnn_layer))*args.hidden_dim, args.hidden_dim))
        u_concatenation_w = nn.Parameter(torch.Tensor(len(eval(args.gnn_layer))*args.hidden_dim, args.hidden_dim))
        init.xavier_uniform_(i_concatenation_w)
        init.xavier_uniform_(u_concatenation_w)
        return i_concatenation_w, u_concatenation_w

    def forward(self, user_embedding_input=None, item_embedding_input=None):
        all_user_embeddings = []
        all_item_embeddings = []
        all_user_embeddingss = []
        all_item_embeddingss = []

        user_embedding = self.user_embedding.weight
        item_embedding = self.item_embedding.weight

        for i, layer in enumerate(self.layers):
            user_embedding, item_embedding, user_embeddings, item_embeddings = layer(user_embedding, item_embedding)
            user_embedding = F.normalize(user_embedding, p=2, dim=1)
            item_embedding = F.normalize(item_embedding, p=2, dim=1)

            all_user_embeddings.append(user_embedding)
            all_item_embeddings.append(item_embedding)
            all_user_embeddingss.append(user_embeddings)
            all_item_embeddingss.append(item_embeddings)

        user_embedding = torch.cat(all_user_embeddings, dim=1)
        item_embedding = torch.cat(all_item_embeddings, dim=1)
        user_embeddings = torch.cat(all_user_embeddingss, dim=2)
        item_embeddings = torch.cat(all_item_embeddingss, dim=2)

        user_embedding = torch.matmul(user_embedding, self.u_concatenation_w)
        item_embedding = torch.matmul(item_embedding, self.i_concatenation_w)
        user_embeddings = torch.matmul(user_embeddings, self.u_concatenation_w)
        item_embeddings = torch.matmul(item_embeddings, self.i_concatenation_w)

        return user_embedding, item_embedding, user_embeddings, item_embeddings


class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, userNum, itemNum, behavior, behavior_mats, matrix_typ):
        super(GCNLayer, self).__init__()
        self.behavior = behavior
        if matrix_typ=='behavior':
            self.behavior_mats = behavior_mats
        if matrix_typ=='composite':
            self.behavior_mats = behavior_mats

        self.userNum = userNum
        self.itemNum = itemNum
        self.user_q_w = nn.Parameter(torch.Tensor(in_dim, in_dim))
        self.user_k_w = nn.Parameter(torch.Tensor(in_dim, in_dim))
        self.user_v_w = nn.Parameter(torch.Tensor(in_dim, in_dim))
        self.item_q_w = nn.Parameter(torch.Tensor(in_dim, in_dim))
        self.item_k_w = nn.Parameter(torch.Tensor(in_dim, in_dim))
        self.item_v_w = nn.Parameter(torch.Tensor(in_dim, in_dim))

        init.xavier_uniform_(self.i_w)
        init.xavier_uniform_(self.u_w)
        init.xavier_uniform_(self.q_w)
        init.xavier_uniform_(self.k_w)
        init.xavier_uniform_(self.v_w)
        self.act = torch.nn.ReLU()
        self.i_w = nn.Parameter(torch.Tensor(in_dim, out_dim))
        self.u_w = nn.Parameter(torch.Tensor(in_dim, out_dim))
        self.trans_weights_s1 = Parameter(
            torch.FloatTensor(len(self.behavior), in_dim, out_dim)
        )
        self.trans_weights_s2 = Parameter(torch.FloatTensor(len(self.behavior),out_dim, 1))
        self.trans_weights_s3 = Parameter(
            torch.FloatTensor(len(self.behavior), in_dim, out_dim)
        )
        self.trans_weights_s4 = Parameter(torch.FloatTensor(len(self.behavior),out_dim, 1))

        init.xavier_uniform_(self.i_w)
        init.xavier_uniform_(self.u_w)
        init.xavier_uniform_(self.user_q_w)
        init.xavier_uniform_(self.user_k_w)
        init.xavier_uniform_(self.user_v_w)
        init.xavier_uniform_(self.item_q_w)
        init.xavier_uniform_(self.item_k_w)
        init.xavier_uniform_(self.item_v_w)
        init.xavier_uniform_(self.user_trans_weights_s1)
        init.xavier_uniform_(self.user_trans_weights_s2)
        init.xavier_uniform_(self.item_trans_weights_s1)
        init.xavier_uniform_(self.item_trans_weights_s2)
        self.reset()

    def reset(self):
        self.trans_weights_s1.data.normal_(std=1.0 / math.sqrt(args.hidden_dim))
        self.trans_weights_s2.data.normal_(std=1.0 / math.sqrt(args.hidden_dim))
        self.trans_weights_s3.data.normal_(std=1.0 / math.sqrt(args.hidden_dim))
        self.trans_weights_s4.data.normal_(std=1.0 / math.sqrt(args.hidden_dim))

    def mh_attention(self, embeddings, q_w, k_w, v_w, trans_weights_s1, trans_weights_s2):
        # Apply QKV attention
        q = torch.matmul(embeddings, q_w)
        k = torch.matmul(embeddings, k_w)
        v = torch.matmul(embeddings, v_w)

        attention_scores = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(self.hidden_dim)

        attention_weights = F.softmax(attention_scores, dim=-1)

        attention_output = torch.bmm(attention_weights, v)
        attention_output = torch.matmul(attention_output, trans_weights_s1)
        attention_output = torch.tanh(attention_output)
        attention_output = torch.matmul(attention_output, trans_weights_s2)
        attention_output = F.softmax(attention_output.squeeze(2), dim=0).unsqueeze(1).permute(2, 1, 0)
        embed_tmp = attention_output.permute(1, 0, 2)
        type_embed = torch.matmul(attention_output, embed_tmp)
        output = embeddings + (torch.matmul(type_embed, v_w).permute(0, 2, 1).squeeze(2))
        output = self.act(output)

        return output
    def forward(self, user_embedding, item_embedding):
        user_embedding_list = [None] * len(self.behavior)
        item_embedding_list = [None] * len(self.behavior)

        for i in range(len(self.behavior)):
            user_embedding_list[i] = torch.spmm(self.behavior_mats[i]['A'], item_embedding)
            item_embedding_list[i] = torch.spmm(self.behavior_mats[i]['AT'], user_embedding)
        user_embeddings = torch.stack(user_embedding_list, dim=0)
        item_embeddings = torch.stack(item_embedding_list, dim=0)
        user_embedding = torch.matmul(torch.mean(user_embeddings, dim=0), self.u_w)
        item_embedding = torch.matmul(torch.mean(item_embeddings, dim=0), self.i_w)

        attention = F.softmax(
            torch.matmul(
                torch.tanh(torch.matmul(user_embeddings, self.trans_weights_s1)), self.trans_weights_s2
            ).squeeze(2),
            dim=0,
        ).unsqueeze(1).permute(2,1,0)
        user_embed_tmp=user_embeddings.permute(1,0,2)
        user_type_embed = torch.matmul(attention, user_embed_tmp)
        user_embed = user_embedding + (torch.matmul(user_type_embed, self.u_w).permute(0,2,1).squeeze(2))

        attention = F.softmax(
            torch.matmul(
                torch.tanh(torch.matmul(item_embeddings, self.trans_weights_s3)), self.trans_weights_s4
            ).squeeze(2),
            dim=0,
        ).unsqueeze(1).permute(2,1,0)
        item_embed_tmp=item_embeddings.permute(1,0,2)
        item_type_embed = torch.matmul(attention, item_embed_tmp)
        item_embed = item_embedding + torch.matmul(item_type_embed, self.i_w).permute(0,2,1).squeeze(2)
        user_embed=self.act(user_embed)
        item_embed=self.act(item_embed)

        user_embeddings = self.act(torch.matmul(user_embeddings, self.u_w))
        item_embeddings = self.act(torch.matmul(item_embeddings, self.i_w))

        return user_embed, item_embed, user_embeddings, item_embeddings


