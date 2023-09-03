import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import models
from models.neus import NeuSModel, VarianceNetwork
from models.utils import chunk_batch
from systems.utils import update_module_step
from nerfacc import ContractionType, OccupancyGrid, ray_marching, render_weight_from_density, render_weight_from_alpha, accumulate_along_rays
from nerfacc.intersection import ray_aabb_intersect

@models.register('bakedsdf')
class BakedSDFModel(NeuSModel):
    def setup(self):
        self.geometry = models.make(self.config.geometry.name, self.config.geometry)
        self.texture = models.make(self.config.texture.name, self.config.texture)
        self.geometry.contraction_type = ContractionType.AABB

        if self.config.learned_background:
            self.geometry_bg = models.make(self.config.geometry_bg.name, self.config.geometry_bg)
            self.texture_bg = models.make(self.config.texture_bg.name, self.config.texture_bg)
            self.geometry_bg.contraction_type = ContractionType.UN_BOUNDED_SPHERE
            self.near_plane_bg, self.far_plane_bg = 0.1, 1e3
            self.cone_angle_bg = 10**(math.log10(self.far_plane_bg) / self.config.num_samples_per_ray_bg) - 1.
            self.render_step_size_bg = 0.01            

        self.variance = VarianceNetwork(self.config.variance)
        self.register_buffer('scene_aabb', torch.as_tensor([-self.config.radius, -self.config.radius, -self.config.radius, self.config.radius, self.config.radius, self.config.radius], dtype=torch.float32))
        if self.config.grid_prune:
            self.occupancy_grid = OccupancyGrid(
                roi_aabb=self.scene_aabb,
                resolution=128,
                contraction_type=ContractionType.AABB
            )
            if self.config.learned_background:
                self.occupancy_grid_bg = OccupancyGrid(
                    roi_aabb=self.scene_aabb,
                    resolution=256,
                    contraction_type=ContractionType.UN_BOUNDED_SPHERE
                )
        self.randomized = self.config.randomized
        self.background_color = None
        self.render_step_size = 1.732 * 2 * self.config.radius / self.config.num_samples_per_ray

    @torch.no_grad()
    def export(self, export_config):
        mesh = self.isosurface()
        if export_config.export_vertex_color:
            _, sdf_grad, feature = chunk_batch(self.geometry, export_config.chunk_size, False, mesh['v_pos'].to(self.rank), with_grad=True, with_feature=True)
            normal = F.normalize(sdf_grad, p=2, dim=-1)
            self.texture.set_shading_mode('diffuse')
            rgb = self.texture(feature, -normal, normal) # set the viewing directions to the normal to get "albedo"
            mesh['v_rgb'] = rgb.cpu()
        return mesh
