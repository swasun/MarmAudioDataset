# Authors: Charly Lamothe <charlylmth_AT_gmail_DOT_com>; Paul Best <paul_DOT_best_AT_lis-lab_DOT_fr>

# License: BSD (3-clause)

import torch
from torch import tensor, nn, exp, log, ones, stack


class PCENLayer(nn.Module):
    def __init__(self, num_bands,
                 s=0.025,
                 alpha=.8,
                 delta=10.,
                 r=.25,
                 eps=1e-6,
                 init_smoother_from_data=True, trainable=True):
        super(PCENLayer, self).__init__()
        self.log_s = nn.Parameter( log(ones((1,1,num_bands)) * s), requires_grad=trainable)
        self.log_alpha = nn.Parameter( log(ones((1,1,num_bands,1)) * alpha), requires_grad=trainable)
        self.log_delta = nn.Parameter( log(ones((1,1,num_bands,1)) * delta), requires_grad=trainable)
        self.log_r = nn.Parameter( log(ones((1,1,num_bands,1)) * r), requires_grad=trainable)
        self.eps = tensor(eps)
        self.init_smoother_from_data = init_smoother_from_data

    def forward(self, input): # expected input (batch, channel, freqs, time)
        init = input[:,:,:,0]  # initialize the filter with the first frame
        if not self.init_smoother_from_data:
            init = torch.zeros(init.shape)  # initialize with zeros instead

        filtered = [init]
        for iframe in range(1, input.shape[-1]):
            filtered.append( (1-exp(self.log_s)) * filtered[iframe-1] + exp(self.log_s) * input[:,:,:,iframe] )
        filtered = stack(filtered).permute(1,2,3,0)

        # stable reformulation due to Vincent Lostanlen; original formula was:
        alpha, delta, r = exp(self.log_alpha), exp(self.log_delta), exp(self.log_r)
        return (input / (self.eps + filtered)**alpha + delta)**r - delta**r
