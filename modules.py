import torch
from torch import nn
import numpy as np
import math
import torch.nn.functional as F
from collections import OrderedDict

# Define BatchLinear without MetaModule
class BatchLinear(nn.Linear):
    '''A linear layer that can deal with batched weight matrices and biases.'''
    
    def forward(self, input, weight=None, bias=None):
        if weight is None:
            return super().forward(input)
        else:
            output = input.matmul(weight.permute(*[i for i in range(len(weight.shape) - 2)], -1, -2))
            if bias is not None:
                output += bias.unsqueeze(-2)
            return output

class Sine(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return torch.sin(30 * input)

class FCBlock(nn.Module):
    '''A fully connected neural network that allows swapping out the weights.'''
    
    def __init__(self, in_features, out_features, num_hidden_layers, hidden_features,
                 outermost_linear=False, nonlinearity='relu', weight_init=None):
        super().__init__()

        self.first_layer_init = None

        # Dictionary of nonlinearities and initializations
        nls_and_inits = {
            'sine': (Sine(), sine_init, first_layer_sine_init, last_layer_sine_init),
            'relu': (nn.ReLU(inplace=True), init_weights_normal, None, None),
            'sigmoid': (nn.Sigmoid(), init_weights_xavier, None, None),
            'tanh': (nn.Tanh(), init_weights_xavier, None, None),
            'selu': (nn.SELU(inplace=True), init_weights_selu, None, None),
            'softplus': (nn.Softplus(), init_weights_normal, None, None),
            'elu': (nn.ELU(inplace=True), init_weights_elu, None, None)
        }

        nl, nl_weight_init, first_layer_init, last_layer_init = nls_and_inits[nonlinearity]

        self.weight_init = weight_init if weight_init is not None else nl_weight_init
        layers = OrderedDict()

        # First layer
        layers['layer0'] = nn.Sequential(
            BatchLinear(in_features, hidden_features), nl
        )

        # Hidden layers
        for i in range(1, num_hidden_layers + 1):
            layers[f'layer{i}'] = nn.Sequential(
                BatchLinear(hidden_features, hidden_features), nl
            )

        # Output layer
        if outermost_linear:
            layers[f'layer{num_hidden_layers + 1}'] = nn.Sequential(
                BatchLinear(hidden_features, out_features)
            )
        else:
            layers[f'layer{num_hidden_layers + 1}'] = nn.Sequential(
                BatchLinear(hidden_features, out_features), nl
            )

        self.net = nn.Sequential(layers)
        if self.weight_init is not None:
            self.net.apply(self.weight_init)

        if first_layer_init is not None:
            self.first_layer_init = first_layer_init
            self.net[0].apply(first_layer_init)

    def forward(self, coords):
        return self.net(coords)

    def get_weights(self):
        return {name: param.clone() for name, param in self.named_parameters()}

    def set_weights(self, weights):
        for name, param in self.named_parameters():
            param.data.copy_(weights[name])
    
    def reset_parameters(self):
        '''Method to reset the parameters of the layers.'''
        if self.weight_init is not None:
            self.net.apply(self.weight_init)

        if self.first_layer_init is not None:
            self.net[0].apply(self.first_layer_init)

class SingleBVPNet(nn.Module):
    '''A canonical representation network for a BVP.'''
    
    def __init__(self, out_features=1, type='sine', in_features=3,
                 hidden_features=256, num_hidden_layers=3):
        super().__init__()
        self.net = FCBlock(in_features=in_features, out_features=out_features,
                           num_hidden_layers=num_hidden_layers, hidden_features=hidden_features,
                           outermost_linear=True, nonlinearity=type)

    def forward(self, model_input):
        coords_org = model_input['coords'].requires_grad_(True)
        output = self.net(coords_org)
        return {'model_in': coords_org, 'model_out': output}
    
    def reset_parameters(self):
        '''Reset the parameters of the network.'''
        self.net.reset_parameters()

# Weight initialization functions
def init_weights_normal(m):
    if isinstance(m, (BatchLinear, nn.Linear)) and hasattr(m, 'weight'):
        nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity='relu')

def init_weights_selu(m):
    if type(m) == BatchLinear or type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            nn.init.normal_(m.weight, std=1 / math.sqrt(num_input))

def init_weights_elu(m):
    if type(m) == BatchLinear or type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            nn.init.normal_(m.weight, std=math.sqrt(1.5505188080679277) / math.sqrt(num_input))

def init_weights_xavier(m):
    if isinstance(m, (BatchLinear, nn.Linear)) and hasattr(m, 'weight'):
        nn.init.xavier_normal_(m.weight)

def sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            m.weight.uniform_(-np.sqrt(6 / num_input) / 30, np.sqrt(6 / num_input) / 30)

def first_layer_sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            m.weight.uniform_(-1 / num_input, 1 / num_input)

def last_layer_sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)