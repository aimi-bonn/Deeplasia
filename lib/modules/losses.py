from torch import nn
import torch


class MultiTaskLoss(nn.Module):
    def __init__(self, sigma_age=1, sigma_sex=1):
        super().__init__()
        self.sigma = nn.Parameter(torch.tensor([sigma_age, sigma_sex]))
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, age_hat, sex_hat, age, sex):
        l = torch.tensor([self.mse(age_hat, age), self.bce(sex_hat, sex)])
        l = 0.5 * torch.Tensor(l) / self.sigma ** 2
        l = l.sum() + torch.log(self.sigma.prod())
        return l
