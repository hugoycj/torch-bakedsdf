import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import trimesh
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

    def forward_(self, rays):
        n_rays = rays.shape[0]
        rays_o, rays_d = rays[:, 0:3], rays[:, 3:6] # both (N_rays, 3)

        with torch.no_grad():
            ray_indices, t_starts, t_ends = ray_marching(
                rays_o, rays_d,
                scene_aabb=self.scene_aabb,
                grid=self.occupancy_grid if self.config.grid_prune else None,
                alpha_fn=None,
                near_plane=None, far_plane=None,
                render_step_size=self.render_step_size,
                stratified=self.randomized,
                cone_angle=0.0,
                alpha_thre=0.0
            )
        
        ray_indices = ray_indices.long()
        t_origins = rays_o[ray_indices]
        t_dirs = rays_d[ray_indices]
        midpoints = (t_starts + t_ends) / 2.
        positions = t_origins + t_dirs * midpoints
        dists = t_ends - t_starts

        if self.config.geometry.grad_type == 'finite_difference':
            sdf, sdf_grad, feature, sdf_laplace = self.geometry(positions, with_grad=True, with_feature=True, with_laplace=True)
        else:
            sdf, sdf_grad, feature = self.geometry(positions, with_grad=True, with_feature=True)
        normal = F.normalize(sdf_grad, p=2, dim=-1)
        alpha = self.get_alpha(sdf, normal, t_dirs, dists)[...,None]
        rgb = self.texture(feature, t_dirs, normal=normal, positions=positions)

        weights = render_weight_from_alpha(alpha, ray_indices=ray_indices, n_rays=n_rays)
        opacity = accumulate_along_rays(weights, ray_indices, values=None, n_rays=n_rays)
        depth = accumulate_along_rays(weights, ray_indices, values=midpoints, n_rays=n_rays)
        comp_rgb = accumulate_along_rays(weights, ray_indices, values=rgb, n_rays=n_rays)

        comp_normal = accumulate_along_rays(weights, ray_indices, values=normal, n_rays=n_rays)
        comp_normal = F.normalize(comp_normal, p=2, dim=-1)

        out = {
            'comp_rgb': comp_rgb,
            'comp_normal': comp_normal,
            'opacity': opacity,
            'depth': depth,
            'rays_valid': opacity > 0,
            'num_samples': torch.as_tensor([len(t_starts)], dtype=torch.int32, device=rays.device)
        }

        if self.training:
            out.update({
                'sdf_samples': sdf,
                'sdf_grad_samples': sdf_grad,
                'weights': weights.view(-1),
                'points': midpoints.view(-1),
                'intervals': dists.view(-1),
                'ray_indices': ray_indices.view(-1)                
            })
            if self.config.geometry.grad_type == 'finite_difference':
                out.update({
                    'sdf_laplace_samples': sdf_laplace
                })

        if self.config.learned_background:
            out_bg = self.forward_bg_(rays)
        else:
            out_bg = {
                'comp_rgb': self.background_color[None,:].expand(*comp_rgb.shape),
                'num_samples': torch.zeros_like(out['num_samples']),
                'rays_valid': torch.zeros_like(out['rays_valid'])
            }

        out_full = {
            'comp_rgb': out['comp_rgb'] + out_bg['comp_rgb'] * (1.0 - out['opacity']),
            'num_samples': out['num_samples'] + out_bg['num_samples'],
            'rays_valid': out['rays_valid'] | out_bg['rays_valid']
        }

        return {
            **out,
            **{k + '_bg': v for k, v in out_bg.items()},
            **{k + '_full': v for k, v in out_full.items()}
        }

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

    @torch.no_grad()
    def export_glb(self, chunk_size=2097152):
        mesh = self.isosurface()
        positions = mesh['v_pos'].to(self.rank)
        _, _, feature = chunk_batch(self.geometry, chunk_size, False, positions, with_grad=True, with_feature=True)
        vertex_colors, attribute_dict = self.texture.get_spherical_gaussian_params(feature, positions)
        tri_mesh = trimesh.Trimesh(vertices=mesh['v_pos'].cpu().numpy(), faces=mesh['t_pos_idx'].cpu().numpy(), \
                                    vertex_colors=vertex_colors)            
        tri_mesh.vertex_attributes.update(attribute_dict)
        return tri_mesh