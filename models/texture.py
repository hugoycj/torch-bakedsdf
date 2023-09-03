import torch
import torch.nn as nn
import numpy as np
import models
from models.utils import get_activation
from models.network_utils import get_encoding, get_mlp, get_encoding_with_network
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

def quantize_and_clip(values):
    values = values * 255
    # values = torch.round(values)
    # values = torch.clamp(values, 0, 255)
    return values

@models.register('volume-color-plus-specular')
class VolumeColorPlusSpecular(nn.Module):
    def __init__(self, config):
        super(VolumeColorPlusSpecular, self).__init__()
        self.config = config
        self.n_output_dims = 3
        self.n_input_dims = self.config.input_feature_dim
        diffuse_network = get_mlp(self.n_input_dims, self.n_output_dims, self.config.mlp_network_config)    
        self.diffuse_network = diffuse_network
        self.diffuse_step = self.config.get('diffuse_step')
        
        self.specular_dim = self.config.get('specular_dim')
        self.sg_blob_num = self.config.get('sg_blob_num')
        self.sepcular_network = get_encoding_with_network(3, 9*self.sg_blob_num, self.config.sg_encoding_config, self.config.sg_network_config)
        self.shading = 'diffuse'

    def set_shading_mode(self, mode):
        self.shading = mode
        
    def forward(self, features, dirs, normal=None, positions=None, *args):
        """
        Args:
            features (torch.Tensor): The input features, in a shape of [N, self.n_input_dims]
            dirs (torch.Tensor): The direction vector, in a shape of [N, 3]
            normal (optional, torch.Tensor): The normal vector, in a shape of [N, 3]
            positions (optional, torch.Tensor): The positions vector, in a shape of [N, 3]
            *args: Other arguments

        Returns:
            torch.Tensor: The output color, in a shape of [N, C]
        """
        _features = features.view(-1, features.shape[-1])
        diffuse = self.diffuse_network(_features).view(*features.shape[:-1], self.n_output_dims).float()
        if 'color_activation' in self.config:
            diffuse = get_activation(self.config.color_activation)(diffuse)
        
        if self.shading == 'diffuse' or positions is None:
            specular = None
            color = diffuse
        else:
            _positions = positions.view(-1, positions.shape[-1])
            specular_feature = self.sepcular_network(positions).reshape((-1, self.sg_blob_num, 9))
            specular = self.spherical_gaussian(dirs, specular_feature)
            color = specular + diffuse # specular + albedo
        return color
    
    def get_spherical_gaussian_params(self, features, positions):
        _features = features.view(-1, features.shape[-1])
        diffuse = self.diffuse_network(_features).view(*features.shape[:-1], self.n_output_dims).float()
        if 'color_activation' in self.config:
            diffuse = get_activation(self.config.color_activation)(diffuse)

        _positions = positions.view(-1, positions.shape[-1])
        lgtSGs = self.sepcular_network(positions).reshape((-1, self.sg_blob_num, 9))
        lgtSGLobes = lgtSGs[..., :3] / (torch.norm(lgtSGs[..., :3], dim=-1, keepdim=True)) # mean, (-1, 1), [N, sg_blob_num, 3]
        lgtSGMus = torch.sigmoid(lgtSGs[..., -3:])  # color, (0, 1) [N, sg_blob_num, 3]
        lgtSGLambdas = torch.abs(lgtSGs[..., 3:4]) #  sgg, (0, 100), [N, sg_blob_num, 3]

        attribute_dict = {}
        cv2gl = torch.tensor([[1.0, 0.0, 0.0],
                                [0.0, 0.0, -1.0],
                                [0.0, -1.0, 0.0]]).to(lgtSGLobes.device)
        lgtSGLobes = torch.matmul(lgtSGLobes, cv2gl)
        for i in range(self.sg_blob_num):
            _lgtSGLobes = (lgtSGLobes[:, i] + 1) / 2
            _lgtSGMus = lgtSGMus[:, i]
            _lgtSGLambdas = lgtSGLambdas[:, i] / 100
            attribute_dict[f'_sg_mean_{i}'] = quantize_and_clip(_lgtSGLobes).cpu().numpy()
            attribute_dict[f'_sg_color_{i}'] = quantize_and_clip(_lgtSGMus).cpu().numpy()
            attribute_dict[f'_sg_scale_{i}'] = quantize_and_clip(_lgtSGLambdas).cpu().numpy()
        vertex_colors = diffuse.cpu().numpy()
        
        return vertex_colors, attribute_dict
    
    def spherical_gaussian(self, viewdirs: torch.Tensor, lgtSGs: torch.Tensor, quantitize=True) -> torch.Tensor:
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
        lgtSGMus = torch.sigmoid(lgtSGs[..., -3:])  # (0, 1), [N, sg_blob_num, 3]
        lgtSGLambdas = torch.abs(lgtSGs[..., 3:4]) #  positive values, [N, sg_blob_num, 3]
        
        #TODO: Fix the quantization error
        if quantitize:
            lgtSGLobes = quantize_and_clip((lgtSGLobes + 1) / 2) * (2.0 / 255)  - 1
            lgtSGMus = quantize_and_clip(lgtSGMus) / 255
            lgtSGLambdas = 100 * quantize_and_clip(lgtSGLambdas / 100) / 255 

        specular = lgtSGMus * torch.exp(lgtSGLambdas * (torch.sum(viewdirs * lgtSGLobes, dim=-1, keepdim=True) - 1.))
        specular = torch.sum(specular, dim=-2)  # [..., 3]
        
        return specular
    
    def update_step(self, epoch, global_step):
        self.shading = 'full' if global_step > self.diffuse_step else 'diffuse' 
    
    def regularizations(self, out):
        return {}