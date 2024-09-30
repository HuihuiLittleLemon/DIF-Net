from torch import nn
import numpy as np
import torch
import modules
import copy

class TransformNetwork(nn.Module):
    def __init__(self, model_type='sine', hidden_num=128, **kwargs):
        super().__init__()

        self.template_field = modules.SingleBVPNet(type=model_type, hidden_features=hidden_num,
                                                   num_hidden_layers=3, in_features=2, out_features=1)
    def forward(self, model_input):
        model_output = self.template_field(model_input)
        # model_output['model_out'] +=  self.W/2
        return model_output

    def reset_parameters(self):
        '''Reset the parameters of the template field'''
        self.template_field.reset_parameters()
    
class NormalNetwork(nn.Module):
    def __init__(self,
                 init_var_theta=np.array([0.0]),
                 init_var_phi=np.array([0.0])):
        super().__init__()

        self.theta = nn.Parameter(torch.from_numpy(init_var_theta), requires_grad=True)
        self.phi = nn.Parameter(torch.from_numpy(init_var_phi), requires_grad=True)
    
    def get_normal(self):
        normal = torch.zeros(self.theta.shape[0],3)
        for i in range(self.theta.shape[0]):
            theta,phi = np.pi*torch.sigmoid(self.theta[i]*2),2*np.pi*torch.sigmoid(self.phi[i]*2)
            normal[i] = torch.tensor([torch.cos(theta)*torch.cos(phi),torch.cos(theta)*torch.sin(phi),torch.sin(theta)])

        return normal


    def forward(self):
        normal = self.get_normal()
        return normal