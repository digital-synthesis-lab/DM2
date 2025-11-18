import torch
from torch_geometric.transforms import BaseTransform


import torch
from torch_geometric.transforms import BaseTransform

class RattleParticles(BaseTransform):
    """Applies a random gaussian noise to particle positions. The standard deviation
    of the noise is drawn uniformly from a range [sigma_min, sigma_max].

    Args:
        sigma_min (float): The minimum standard deviation of the Gaussian noise.
        sigma_max (float): The maximum standard deviation of the Gaussian noise.
    """
    def __init__(self, sigma_max, sigma_min=0.001):
        super().__init__()     # ⭐ MUST HAVE
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
    
    def __call__(self, data):
        # Random noise magnitude
        if data.batch is not None:
            sigma = torch.empty(data.num_graphs, device=data.pos.device).uniform_(
                self.sigma_min, self.sigma_max
            )
            sigma = sigma[data.batch, None]
        else:
            sigma = torch.empty(1, device=data.pos.device).uniform_(
                self.sigma_min, self.sigma_max
            )
        
        # Apply perturbation
        eps = torch.randn_like(data.pos)
        data.dx = sigma * eps
        data.pos = data.pos + data.dx

        # Update edge vectors
        if data.edge_attr is not None:
            i, j = data.edge_index
            data.edge_attr = data.edge_attr + data.dx[j] - data.dx[i]

        # Save auxiliary info
        data.sigma = sigma
        data.eps = eps
        return data
    
    def forward(self, data):   # ⭐ MUST HAVE
        return self.__call__(data)



class RattleParticles_TimeEmbedded(BaseTransform):
    """Applies a random gaussian noise to particle positions, scaled by time.
    
    Args:
        sigma_min (float): The minimum standard deviation of the Gaussian noise.
        sigma_max (float): The maximum standard deviation of the Gaussian noise.
    """
    def __init__(self, sigma_max, sigma_min=0.001):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
    
    def __call__(self, data, t):
        # If `data` is a batch, apply noises of different magnitudes to the individual samples
        if data.batch is not None:
            sigma = torch.empty(data.num_graphs, device=data.pos.device).uniform_(self.sigma_min, self.sigma_max)
            sigma = sigma[data.batch, None]
        else:
            sigma = torch.empty(1, device=data.pos.device).uniform_(self.sigma_min, self.sigma_max)
        
        # Add time-scaled noise
        eps = torch.randn_like(data.pos)
        data.dx = sigma * eps * t  # Scale noise by time parameter
        data.pos += data.dx
        
        # If `data` has edge vectors `edge_attr`, update them as well
        if data.edge_attr is not None:
            i, j = data.edge_index
            data.edge_attr += data.dx[j] - data.dx[i]
            
        # Store useful quantities
        data.sigma = sigma
        data.eps = eps
        
        return data
    
    def __repr__(self):
        return f'{self.__class__.__name__}(sigma_min={self.sigma_min}, sigma_max={self.sigma_max})'
