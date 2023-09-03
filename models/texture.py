import torch
import torch.nn as nn

import models
from models.utils import get_activation
from models.network_utils import get_encoding, get_mlp
from systems.utils import update_module_step


@models.register('volume-radiance')
class VolumeRadiance(nn.Module):
    def __init__(self, config):
        super(VolumeRadiance, self).__init__()
        self.config = config
        self.n_dir_dims = self.config.get('n_dir_dims', 3)
        self.n_output_dims = 3
        encoding = get_encoding(self.n_dir_dims, self.config.dir_encoding_config)
        self.n_input_dims = self.config.input_feature_dim + encoding.n_output_dims
        network = get_mlp(self.n_input_dims, self.n_output_dims, self.config.mlp_network_config)    
        self.encoding = encoding
        self.network = network
    
    def forward(self, features, dirs, *args):
        dirs = (dirs + 1.) / 2. # (-1, 1) => (0, 1)
        dirs_embd = self.encoding(dirs.view(-1, self.n_dir_dims))
        network_inp = torch.cat([features.view(-1, features.shape[-1]), dirs_embd] + [arg.view(-1, arg.shape[-1]) for arg in args], dim=-1)
        color = self.network(network_inp).view(*features.shape[:-1], self.n_output_dims).float()
        if 'color_activation' in self.config:
            color = get_activation(self.config.color_activation)(color)
        return color

    def update_step(self, epoch, global_step):
        update_module_step(self.encoding, epoch, global_step)

    def regularizations(self, out):
        return {}


@models.register('volume-color')
class VolumeColor(nn.Module):
    def __init__(self, config):
        super(VolumeColor, self).__init__()
        self.config = config
        self.n_output_dims = 3
        self.n_input_dims = self.config.input_feature_dim
        network = get_mlp(self.n_input_dims, self.n_output_dims, self.config.mlp_network_config)
        self.network = network
    
    def forward(self, features, *args):
        network_inp = features.view(-1, features.shape[-1])
        color = self.network(network_inp).view(*features.shape[:-1], self.n_output_dims).float()
        if 'color_activation' in self.config:
            color = get_activation(self.config.color_activation)(color)
        return color

    def regularizations(self, out):
        return {}

@models.register('volume-color-plus-specular')
class VolumeColorPlusSpecular(nn.Module):
    def __init__(self, config):
        super(VolumeColorPlusSpecular, self).__init__()
        self.config = config
        self.specular_dim = self.config.get('specular_dim')
        self.n_output_dims = 3 + self.specular_dim
        self.n_input_dims = self.config.input_feature_dim
        diffuse_network = get_mlp(self.n_input_dims, self.n_output_dims, self.config.mlp_network_config)    
        self.diffuse_network = diffuse_network
        self.diffuse_step = self.config.get('diffuse_step')
        
        self.sg_blob_num = self.config.get('sg_blob_num')
        sepcular_network = get_mlp(self.specular_dim, 9*self.sg_blob_num, self.config.mlp_network_config)
        self.sepcular_network = sepcular_network
        self.shading = 'diffuse'

    def forward(self, features, dirs, *args):
        """
        Args:
            features (torch.Tensor): The input features, in a shape of [N, self.n_input_dims]
            dirs (torch.Tensor): The direction vector, in a shape of [N, 3]
            *args: Other arguments

        Returns:
            torch.Tensor: The output color, in a shape of [N, C]
        """
        network_inp = features.view(-1, features.shape[-1])
        color_feature = self.diffuse_network(network_inp).view(*features.shape[:-1], self.n_output_dims).float()
        diffuse = color_feature[..., :3]
        if 'color_activation' in self.config:
            diffuse = get_activation(self.config.color_activation)(diffuse)
        
        if self.shading == 'diffuse':
            specular = None
            color = diffuse
        else:
            specular_feature = self.sepcular_network(color_feature[..., 3:]).reshape((-1, self.sg_blob_num, 9))
            specular = self.spherical_gaussian(dirs, specular_feature)
            color = specular + diffuse # specular + albedo
        return diffuse
    
    def spherical_gaussian(self, viewdirs: torch.Tensor, lgtSGs: torch.Tensor) -> torch.Tensor:
        """
        Calculate the specular component of a Spherical Gaussian (SG) model.

        Args:
            direction (torch.Tensor): The direction vector, in a shape of [N, 3]
            lgtSGs (torch.Tensor): The parameter of the SGs, in a shape of [N, sg_blob_num, 7]

        Returns:
            torch.Tensor: The specular component, in a shape of [N, 3]
        """
        
        viewdirs = viewdirs.unsqueeze(-2)  # [..., 1, 3]
        
        lgtSGLobes = lgtSGs[..., :3] / (torch.norm(lgtSGs[..., :3], dim=-1, keepdim=True)) # (-1, 1), [N, sg_blob_num, 3]
        lgtSGLambdas = torch.abs(lgtSGs[..., 3:4]) #  positive values, [N, sg_blob_num, 3]
        lgtSGMus = torch.sigmoid(lgtSGs[..., -3:])  # (0, 1), [N, sg_blob_num, 3]
        
        specular = lgtSGMus * torch.exp(lgtSGLambdas * (torch.sum(viewdirs * lgtSGLobes, dim=-1, keepdim=True) - 1.))
        specular = torch.sum(specular, dim=-2)  # [..., 3]
        
        return specular
    
    def update_step(self, epoch, global_step):
        self.shading = 'diffuse' if global_step > self.diffuse_step else 'full'
    
    def regularizations(self, out):
        return {}