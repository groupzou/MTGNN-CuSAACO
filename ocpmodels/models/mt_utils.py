import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn.inits import glorot_orthogonal
import math


class DeepDockMDN(nn.Module):
    def __init__(self, emb_size, n_res_layer=3, n_gaussians=20):
        super(DeepDockMDN, self).__init__()
        self.n_gaussians = n_gaussians

        mdn_layer = [nn.Linear(emb_size, int(emb_size / 2)), nn.BatchNorm1d(int(emb_size / 2)), nn.ReLU(),
                     nn.Linear(int(emb_size / 2), 1)]
        self.mdn_layer = nn.Sequential(*mdn_layer)
        self.pi = nn.Linear(1, n_gaussians)
        self.sigma = nn.Linear(1, n_gaussians)
        self.mu = nn.Linear(1, n_gaussians)
        self.reset_params()

    def reset_params(self):
        nn.init.constant_(self.pi.weight, 0.01)
        nn.init.uniform_(self.sigma.weight, -0.005, 0.005)
        nn.init.normal_(self.mu.weight, std=0.01)

    def forward(self, d_pred):
        # w1 = self.pi.weight.detach().numpy()
        # w2 = self.sigma.weight.detach().numpy()
        # w3 = self.mu.weight.detach().numpy()

        d_pred = self.mdn_layer(d_pred)
        # d_pred_np = d_pred.detach().numpy()

        pi = self.pi(d_pred)
        # pi_np1 = pi.detach().numpy()
        pi = F.softmax(torch.clamp(pi, min=1e-8, max=1.), -1)
        # pi_np2 = pi.detach().numpy()

        sigma = self.sigma(d_pred)
        # sigma_np1 = sigma.detach().numpy()
        sigma = F.elu(sigma) + 1.3
        # sigma_np2 = sigma.detach().numpy()

        mu = self.mu(d_pred)

        return pi, sigma, mu


class EdgePred(nn.Module):
    def __init__(self, emb_size):
        super(EdgePred, self).__init__()
        edge_layer = [nn.Linear(emb_size, int(emb_size / 2)), nn.BatchNorm1d(int(emb_size / 2)), nn.ReLU(),
                      nn.Linear(int(emb_size / 2), 1)]
        # self.edge_layer = nn.Sequential(*edge_layer)
        self.edge_layer = nn.ModuleList(edge_layer)

    def forward(self, d_pred):
        # d_pred = self.edge_layer(d_pred)
        for layer in self.edge_layer:
            d_pred = layer(d_pred)
        return d_pred

#     def reset_params(self):
#         for lin in self.edge_layer:
#             glorot_orthogonal(lin.weight, scale=2.0)
#             lin.bias.data.fill_(0)
#         return


def log_sum_exp(x, dim=1):
    
    """ Source: https://github.com/axelbrando/Mixture-Density-Networks-for-distribution-and-uncertainty-estimation; 
        Log-sum-exp trick implementation """
    
    x_max = torch.max(x, dim=dim, keepdim=True)[0]
    return torch.log(torch.sum(torch.exp(x - x_max), dim=dim, keepdim=True)) + x_max


def mean_log_Gaussian_like(alpha, sigma, mu, y_true):

    """ Source: https://github.com/axelbrando/Mixture-Density-Networks-for-distribution-and-uncertainty-estimation """
    
    y_true = y_true.expand_as(mu)
    exponent = torch.log(alpha) - 0.5 * math.log((2 * math.pi)) \
               - torch.log(sigma) \
               - (y_true - mu) ** 2 / (2 * sigma ** 2)

    log_gauss = log_sum_exp(exponent, dim=1)
    res = - torch.mean(log_gauss)
    return res


class CrossStitchUnit(nn.Module):
    def __init__(self, emb_size):
        super(CrossStitchUnit, self).__init__()
        self.emb_size = emb_size
        self.csu = nn.Linear(2 * emb_size, 2 * emb_size, bias=False)

    def reset_parameters(self):
        nn.init.eye_(self.csu.weight)

    def forward(self, input_1, input_2):
        mixed = torch.cat([input_1, input_2], dim=-1)
        mixed = self.csu(mixed)
        out_1, out_2 = torch.split(mixed, [self.emb_size, self.emb_size], dim=-1)
        return out_1, out_2
