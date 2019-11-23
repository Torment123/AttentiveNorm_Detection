import torch
import torch.nn as nn
import torch.nn.functional as F


class hsigmoid(nn.Module):
    def forward(self, x):
        out = F.relu6(x + 3, inplace=True) / 6
        return out


class AttentionWeights(nn.Module):
    def __init__(self, num_channels, k, attention_mode=0):
        super(AttentionWeights, self).__init__()
        self.k = k
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        layers = []
        if attention_mode == 0:
            layers = [ nn.Conv2d(num_channels, k, 1),
                       hsigmoid() ]
        elif attention_mode == 2:
            layers = [ nn.Conv2d(num_channels, k, 1, bias=False),
                        hsigmoid() ]
        else:
            raise NotImplementedError("Unknow attention weight type")
        self.attention = nn.Sequential(*layers)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avgpool(x)#.view(b, c)
        var = torch.var(x, dim=(2, 3)).view(b, c, 1, 1)
        y *= (var + 1e-3).rsqrt()
        #y = torch.cat((y, var), dim=1)
        return self.attention(y).view(b, self.k)



class AttentiveGroupNorm(nn.GroupNorm):
    def __init__(self, num_groups, num_channels, k, eps=1e-5):
        super(AttentiveGroupNorm, self).__init__(num_groups, num_channels, eps=eps, affine=False)
        self.k = k
        self.weight_ = nn.Parameter(torch.Tensor(k, num_channels))
        self.bias_ = nn.Parameter(torch.Tensor(k, num_channels))

        self.attention_weights = AttentionWeights(num_channels, k)

        self._init_params()

    def _init_params(self):
        nn.init.normal_(self.weight_, 1, 0.1)
        nn.init.normal_(self.bias_, 0, 0.1)

    def forward(self, input):
        output = super(AttentiveGroupNorm, self).forward(input)
        size = output.size()
        y = self.attention_weights(input)

        weight = y @ self.weight_
        bias = y @ self.bias_
        weight = weight.unsqueeze(-1).unsqueeze(-1).expand(size)
        bias = bias.unsqueeze(-1).unsqueeze(-1).expand(size)

        return weight * output + bias
