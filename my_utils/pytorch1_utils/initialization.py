import math

import torch
import torch.nn.init as init


def kaiming_normal_(tensor, gain=math.sqrt(2), mode='fan_in'):
    fan = init._calculate_correct_fan(tensor, mode)
    std = gain / math.sqrt(fan)
    with torch.no_grad():
        return tensor.normal_(0, std)


def kaiming_uniform_(tensor, gain=math.sqrt(2), mode='fan_in'):
    fan = init._calculate_correct_fan(tensor, mode)
    std = gain / math.sqrt(fan)
    bound = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
    with torch.no_grad():
        return tensor.uniform_(-bound, bound)