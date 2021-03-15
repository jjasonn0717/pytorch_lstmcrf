import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()

    def forward(self, title_output, attr_output):
        '''
        title_output (batchsize, seqlen, hidden_dim)
        attr_output (batchsize, hidden_dim)
        '''
        seq_len = title_output.size()[1]
        attr_output = attr_output.unsqueeze(1).repeat(1,seq_len,1)
        cos_sim = torch.cosine_similarity(attr_output,title_output,-1)
        cos_sim = cos_sim.unsqueeze(-1)
        outputs = title_output * cos_sim
        return outputs
