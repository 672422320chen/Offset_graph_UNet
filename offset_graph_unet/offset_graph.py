import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



class GraphUnet(nn.Module):

    def __init__(self, ks, in_dim, out_dim, dim, drop_p,adjacency_matrix):
        super(GraphUnet, self).__init__()

     
        self.ks = ks 
        self.bottom_gcn = GCN(dim, dim,  drop_p,adjacency_matrix)
        self.down_gcns = nn.ModuleList()
        self.up_gcns = nn.ModuleList()
        self.off_gcns = nn.ModuleList()
        
 
        self.pools = nn.ModuleList()
        
        self.unpools = nn.ModuleList()
        
    
        
        self.l_n = len(ks)
        
        
        for i in range(self.l_n):
            self.down_gcns.append(GCN(dim, dim,  drop_p, adjacency_matrix))
            self.up_gcns.append(GCN(dim, dim,  drop_p, adjacency_matrix))
            self.off_gcns.append(GCN11(dim, dim, drop_p))
            self.pools.append(Pool(ks[i], dim, drop_p))
            self.unpools.append(Unpool(dim, dim, drop_p))        
     

    def forward(self, g, h):
        adj_ms = []
        indices_list = [] 
        down_outs = [] 
        hs = [] 
        org_h = h 
        
       
        for i in range(self.l_n):
            h = self.down_gcns[i](g, h) 
            adj_ms.append(g) 
            down_outs.append(h) 
            g, h, idx = self.pools[i](g, h) 
            
            indices_list.append(idx) 
        
        h = self.bottom_gcn(g, h) 
        
        for i in range(self.l_n):
            up_idx = self.l_n - i - 1 
            g, idx = adj_ms[up_idx], indices_list[up_idx] 
            g, h = self.unpools[i](g, h, down_outs[up_idx], idx) 
            off = h - down_outs[up_idx] 
            off_w = self.off_gcns[i](off) 
            off_f = off_w * h 
            h = off_f.add(down_outs[up_idx]) 
            h = self.up_gcns[i](g, h) 
          
        h = h.add(org_h) 
       
        return h, hs, g




class GCN(nn.Module):

    def __init__(self, in_dim, out_dim, p,adjacency_matrix):
        super(GCN, self).__init__()
        self.proj = nn.Linear(in_dim, out_dim)
        self.BN = nn.BatchNorm1d(in_dim)
        self.drop = nn.Dropout(p=p) if p > 0.0 else nn.Identity()
        self.act = nn.LeakyReLU(inplace=True)
        self.adj = adjacency_matrix
        self.I = torch.eye(adjacency_matrix.shape[0], adjacency_matrix.shape[0], requires_grad=False, device=device, dtype=torch.float32)
        self.mask = torch.ceil(adjacency_matrix * 0.00001)
        self.lambda_ = nn.Parameter(torch.zeros(1))

    def A_to_D_inv(self, g):
        D = g.sum(1)
        D_hat = torch.diag(torch.pow(D, -0.5))
        return D_hat
        
    def forward(self, g, h):
        
        h = self.BN(h)
        g = g + torch.eye(g.shape[0], g.shape[0], requires_grad=False, device=device, dtype=torch.float32)
        D_hat = self.A_to_D_inv(g)
        g = torch.matmul(D_hat, torch.matmul(g,D_hat))
        h = self.proj(h)
        h = torch.matmul(g, h)
        
        h = self.act(h)
        return h




class GCN11(nn.Module):

    def __init__(self, in_dim, out_dim, p):
        super(GCN11, self).__init__()
        self.proj = nn.Linear(in_dim, out_dim)
        self.BN = nn.BatchNorm1d(in_dim)
        self.drop = nn.Dropout(p=p) if p > 0.0 else nn.Identity()
        self.act1 = nn.Sigmoid()
        self.act2 = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        x = self.BN(x)
        x = self.proj(x)
        x = self.drop(x)
        x = self.act2(x)
        return x






class Pool(nn.Module):

    def __init__(self, k, in_dim, p):
        super(Pool, self).__init__()
        self.k = k 
        self.sigmoid = nn.Sigmoid()
        self.proj = nn.Linear(in_dim, 1)
        self.drop = nn.Dropout(p=p) if p > 0 else nn.Identity()

    def forward(self, g, h):
        Z = self.drop(h) 
        weights = self.proj(Z).squeeze() 
        scores = self.sigmoid(weights)
        return top_k_graph(scores, g, h, self.k)
      

class Unpool(nn.Module):

    def __init__(self, *args):
        super(Unpool, self).__init__()

    def forward(self, g, h, pre_h, idx):
        new_h = h.new_zeros([g.shape[0], h.shape[1]])
        new_h[idx] = h
        return g, new_h


def top_k_graph(scores, g, h, k):
    num_nodes = g.shape[0]
    values, idx = torch.topk(scores, max(2, int(k*num_nodes)))
    new_h = h[idx, :]
    
    values = torch.unsqueeze(values, -1)
    new_h = torch.mul(new_h, values)
    un_g = g.bool().float()
    un_g = torch.matmul(un_g, un_g).bool().float()
    un_g = un_g[idx, :] 
    un_g = un_g[:, idx]
    g = norm_g(un_g)
    return g, new_h, idx


def norm_g(g):
    degrees = torch.sum(g, 1)
    g = g / degrees
    return g




class Initializer(object):

    @classmethod
    def _glorot_uniform(cls, w):
        if len(w.size()) == 2:
            fan_in, fan_out = w.size()
        elif len(w.size()) == 3:
            fan_in = w.size()[1] * w.size()[2]
            fan_out = w.size()[0] * w.size()[2]
        else:
            fan_in = np.prod(w.size())
            fan_out = np.prod(w.size())
        limit = np.sqrt(6.0 / (fan_in + fan_out))
        w.uniform_(-limit, limit)

    @classmethod
    def _param_init(cls, m):
        if isinstance(m, nn.parameter.Parameter):
            cls._glorot_uniform(m.data)
        elif isinstance(m, nn.Linear):
            m.bias.data.zero_()
            cls._glorot_uniform(m.weight.data)

    @classmethod
    def weights_init(cls, m):
        for p in m.modules():
            if isinstance(p, nn.ParameterList):
                for pp in p:
                    cls._param_init(pp)
            else:
                cls._param_init(p)

        for name, p in m.named_parameters():
            if '.' not in name:
                cls._param_init(p)



class SSConv(nn.Module):
    '''
    Spectral-Spatial Convolution
    '''
    def __init__(self, in_ch, out_ch,kernel_size=3):
        super(SSConv, self).__init__()
        self.depth_conv = nn.Conv2d(
            in_channels=out_ch//2,
            out_channels=out_ch,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size//2,
            
        )
        self.point_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch//2,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            bias=False
        )
        self.Act1 = nn.LeakyReLU(inplace=True)
        self.Act2 = nn.LeakyReLU(inplace=True)
        self.BN=nn.BatchNorm2d(in_ch)
        
    
    def forward(self, input):
        out = self.point_conv(self.BN(input))
        out = self.Act1(out)
        out = self.depth_conv(out)
        out = self.Act2(out)
        return out


class offset_graph(nn.Module):
    def __init__(self, height: int, width: int, changel: int, class_count: int, Q: torch.Tensor, A: torch.Tensor, model='normal'):
        super(offset_graph, self).__init__()
        
        self.class_count = class_count  
        
        self.channel = changel
        self.height = height
        self.width = width
        self.Q = Q
        self.A = A
        self.model=model
        self.norm_col_Q = Q / (torch.sum(Q, 0, keepdim=True)) 
        self.ks = torch.tensor([0.95,0.9]) 
       
        self.act = nn.LeakyReLU(inplace=True)
        Initializer.weights_init(self)
        nodes_count=self.A.shape[0]
        self.I = torch.eye(nodes_count, nodes_count, requires_grad=False).to(device)
        layers_count=2
        
        self.CNN_denoise = nn.Sequential()
        for i in range(layers_count):
            if i == 0:
                self.CNN_denoise.add_module('CNN_denoise_BN'+str(i),nn.BatchNorm2d(self.channel))
                self.CNN_denoise.add_module('CNN_denoise_Conv'+str(i),nn.Conv2d(self.channel, 30, kernel_size=(1, 1)))
                self.CNN_denoise.add_module('CNN_denoise_Act'+str(i),nn.LeakyReLU())
            else:
                self.CNN_denoise.add_module('CNN_denoise_BN'+str(i),nn.BatchNorm2d(30),)
                self.CNN_denoise.add_module('CNN_denoise_Conv' + str(i), nn.Conv2d(30, 30, kernel_size=(1, 1)))
                self.CNN_denoise.add_module('CNN_denoise_Act' + str(i), nn.LeakyReLU())
        
        self.s_gcn = GCN(30, 30, 0.0, self.A) 
        self.g_unet = GraphUnet(self.ks, 30, 30, 30,  0.0, self.A)
        self.l_gcn = GCN(30, 30, 0.0, self.A)
        self.conv2 = SSConv(30, 30, kernel_size=5)
        self.conv3 = SSConv(30, 30, kernel_size=5)
        self.conv4 = SSConv(30, 30, kernel_size=5)

        
        self.Softmax_linear =nn.Sequential(nn.Linear(30, self.class_count)) #IP:
    
    
    def norm_g(g):
        degrees = torch.sum(g, 1)
        g = g / degrees 
        return g

    def forward(self, x: torch.Tensor,showFlag=False):
       
        (hei, wid, c) = x.shape 
        noise = self.CNN_denoise(torch.unsqueeze(x.permute([2, 0, 1]), 0))
        noise =torch.squeeze(noise, 0).permute([1, 2, 0])
        clean_x=noise
        clean_x_flatten = clean_x.reshape([hei * wid, -1]) 
        superpixels_flatten = torch.mm(self.norm_col_Q.t(), clean_x_flatten)  
        H = superpixels_flatten 
        A_ = norm_g(self.A) 
        H = self.s_gcn(A_, H)
        h, hs, g = self.g_unet(A_, H) 
        hs = self.l_gcn(g, h+superpixels_flatten)   
        fuse_feature = hs

        GCN_result = torch.matmul(self.Q, fuse_feature)  
        GCN_result = GCN_result.reshape([hei, wid, -1]) 
        GCN_result = torch.unsqueeze(GCN_result.permute([2, 0, 1]), 0)
        cnn1 = self.conv2(GCN_result)
        res1 = cnn1 + GCN_result
        cnn2 = self.conv3(res1)
        res2 = res1 + cnn2
        cnn3 = self.conv4(res2) 
        out = res2 + cnn3
        out = torch.squeeze(out, 0).permute([1, 2, 0]).reshape([hei * wid, -1])
        Y = self.Softmax_linear(out)
        Y = F.softmax(Y, -1) 
        
        return Y