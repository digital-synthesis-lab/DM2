import torch
import torch.nn as nn
import torch.nn.functional as F

from e3nn       import o3
from e3nn.nn    import Gate

from ..conv.e3nn_nequip import Interaction

def tp_path_exists(irreps_in1, irreps_in2, ir_out):
    irreps_in1 = o3.Irreps(irreps_in1).simplify()
    irreps_in2 = o3.Irreps(irreps_in2).simplify()
    ir_out = o3.Irrep(ir_out)

    for _, ir1 in irreps_in1:
        for _, ir2 in irreps_in2:
            if ir_out in ir1 * ir2:
                return True
    return False


class Compose(nn.Module):
    def __init__(self, first, second):
        super().__init__()
        self.first = first
        self.second = second
        self.irreps_in = self.first.irreps_in
        self.irreps_out = self.second.irreps_out

    def forward(self, *input):
        x = self.first(*input)
        return self.second(x)

class FourierFeatureEmbedding(nn.Module):
    """
    Fourier feature embedding for scalar values, which can better represent periodic patterns 
    and capture multiple frequency information in the cooling rate.
    
    Args:
        embedding_dim: Dimension of the output embedding vector (must be even)
        scale: Controls the frequency scale of the Fourier features
        min_value: Minimum expected value for normalization
        max_value: Maximum expected value for normalization
    """
    def __init__(
        self, 
        embedding_dim=32, 
        scale=10.0,
        min_value=-4.0,  # log10(0.0001)
        max_value=4.0,   # log10(10000)
    ):
        super().__init__()
        assert embedding_dim % 2 == 0, "embedding_dim must be even"
        
        self.embedding_dim = embedding_dim
        self.scale = scale
        self.min_value = min_value
        self.max_value = max_value
        
        # Create frequency bands for the Fourier features
        # Using a geometric sequence for better coverage of frequencies
        self.freq_bands = torch.exp(
            torch.linspace(0, torch.log(torch.tensor(scale)), embedding_dim // 2)
        )
        self.register_buffer('frequencies', self.freq_bands)
        
        # Optional projection layer to mix the Fourier features
        self.projection = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
        
    def forward(self, x):
        """
        Forward pass of the Fourier embedding.
        
        Args:
            x: Input tensor of shape [batch_size] or [batch_size, 1]
               with log-transformed cooling rate values
               
        Returns:
            Embedded representation of shape [batch_size, embedding_dim]
        """
        # Ensure x is 2D: [batch_size, 1]
        if x.dim() == 1:
            x = x.unsqueeze(1)
            
        # Normalize to [-1, 1] range for more stable Fourier features
        x_normalized = 2 * (x - self.min_value) / (self.max_value - self.min_value) - 1
        
        # Calculate Fourier features
        # frequencies shape: [embedding_dim//2]
        # x_expanded shape: [batch_size, 1] -> [batch_size, embedding_dim//2]
        x_expanded = x_normalized * self.frequencies.view(1, -1)
        
        # Apply sine and cosine to get the Fourier features
        # Output shape: [batch_size, embedding_dim]
        fourier_features = torch.cat([torch.sin(x_expanded), torch.cos(x_expanded)], dim=-1)
        
        # Apply optional projection
        embedding = self.projection(fourier_features)
        
        return embedding


class AdaptiveLayerModulation(nn.Module):
    """
    Creates layer-specific modulations based on cooling rate. Instead of adding the same 
    embedding to each layer, this creates unique transformations for each layer.
    
    Args:
        base_embed: Base embedding module (Gaussian or Fourier)
        num_layers: Number of graph convolution layers to modulate
        feature_dim: Feature dimension for each layer
    """
    def __init__(
        self,
        base_embed,
        num_layers=3,
        feature_dim=128,
        compression_factor=4  # Higher = less memory
    ):
        super().__init__()
        self.base_embed = base_embed
        self.num_layers = num_layers
        
        # Get base dimension
        base_dim = base_embed.layer2.out_features if hasattr(base_embed, 'layer2') else base_embed.embedding_dim
        
        # Shared dimension reduction
        self.shared_reduction = nn.Sequential(
            nn.Linear(base_dim, base_dim // compression_factor),
            nn.SiLU()
        )
        
        # Lightweight layer-specific projections
        reduced_dim = base_dim // compression_factor
        self.layer_projections = nn.ModuleList([
            nn.Linear(reduced_dim, feature_dim)
            for _ in range(num_layers)
        ])
    
    def forward(self, x):
        # Get base embedding and reduce dimension once
        base_embedding = self.base_embed(x)
        reduced_embedding = self.shared_reduction(base_embedding)
        
        # Apply layer-specific projections to the reduced embedding
        return [proj(reduced_embedding) for proj in self.layer_projections]


class ScaleAndShiftModulation(nn.Module):
    """
    Instead of simply adding the cooling rate embedding to the node features,
    this module learns to scale and shift the features based on the cooling rate.
    This is inspired by Feature-wise Linear Modulation (FiLM) and AdaIN techniques.
    
    Args:
        base_embed: Base embedding module
        feature_dim: Feature dimension to modulate
    """
    def __init__(self, base_embed, feature_dim=128):
        super().__init__()
        self.base_embed = base_embed
        base_dim = base_embed.layer2.out_features if hasattr(base_embed, 'layer2') else base_embed.embedding_dim
        
        # Generate scale and shift parameters
        self.scale_projection = nn.Sequential(
            nn.Linear(base_dim, base_dim),
            nn.SiLU(),
            nn.Linear(base_dim, feature_dim),
            nn.Sigmoid()  # Keep scales in [0, 1] range and then adjust
        )
        
        self.shift_projection = nn.Sequential(
            nn.Linear(base_dim, base_dim),
            nn.SiLU(),
            nn.Linear(base_dim, feature_dim)
        )
    
    def forward(self, x, features):
        """
        Apply modulation to input features.
        
        Args:
            x: Cooling rate tensor
            features: Node features to modulate
            
        Returns:
            Modulated features
        """
        base_embedding = self.base_embed(x)
        
        # Generate scale and shift parameters
        # Scale parameters are adjusted to be centered around 1
        scale = 0.5 + self.scale_projection(base_embedding)  # Range [0.5, 1.5]
        shift = self.shift_projection(base_embedding)
        
        # Expand to match batch dimension of features if needed
        if scale.size(0) != features.size(0):
            batch_assignment = getattr(features, 'batch', torch.zeros(features.size(0), device=features.device, dtype=torch.long))
            scale = scale[batch_assignment]
            shift = shift[batch_assignment]
        
        # Apply scale and shift: y = scale * x + shift
        return scale * features + shift


class HierarchicalCoolingRateEmbedding(nn.Module):
    """
    Instead of using a single embedding, this approach uses multiple embeddings at
    different frequency scales to better capture the hierarchical effects of cooling rates.
    
    Args:
        embedding_dim: Dimension of the output embedding vector
        num_levels: Number of hierarchical levels
    """
    def __init__(
        self,
        embedding_dim=32,
        num_levels=3,
        min_value=-4.0,  # log10(0.0001)
        max_value=4.0,   # log10(10000)
    ):
        super().__init__()
        self.num_levels = num_levels
        self.embedding_dim = embedding_dim
        
        # Create multiple Gaussian basis embeddings with different sigmas
        self.embeddings = nn.ModuleList([
            GaussianBasisEmbedding(
                num_basis=12, 
                embedding_dim=embedding_dim // num_levels,
                min_sigma=0.1 * (2 ** level),  # Wider basis functions for higher levels
                min_value=min_value,
                max_value=max_value
            )
            for level in range(num_levels)
        ])
        
        # Projection to combine all levels
        self.projection = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
    
    def forward(self, x):
        """
        Forward pass of the hierarchical embedding.
        
        Args:
            x: Input tensor of cooling rates
            
        Returns:
            Combined hierarchical embedding
        """
        # Get embeddings from all levels
        level_embeddings = [embed(x) for embed in self.embeddings]
        
        # Concatenate all embeddings
        combined = torch.cat(level_embeddings, dim=-1)
        
        # Apply projection for feature mixing
        return self.projection(combined)


class CrossAttentionEmbedding(nn.Module):
    """
    This embedding uses cross-attention between cooling rate embeddings and node features
    to dynamically adjust how the cooling rate influences different nodes based on their features.
    
    Args:
        base_embed: Base embedding module for cooling rate
        node_dim: Dimension of node features
        num_heads: Number of attention heads
    """
    def __init__(
        self,
        base_embed,
        node_dim=128,
        num_heads=4
    ):
        super().__init__()
        self.base_embed = base_embed
        base_dim = base_embed.layer2.out_features if hasattr(base_embed, 'layer2') else base_embed.embedding_dim
        
        self.cooling_rate_projection = nn.Linear(base_dim, node_dim)
        
        # Multi-head cross-attention
        self.norm1 = nn.LayerNorm(node_dim)
        self.norm2 = nn.LayerNorm(node_dim)
        self.attn = nn.MultiheadAttention(node_dim, num_heads, batch_first=True)
        
        # Feed-forward network
        self.ff = nn.Sequential(
            nn.Linear(node_dim, node_dim * 4),
            nn.SiLU(),
            nn.Linear(node_dim * 4, node_dim)
        )
    
    def forward(self, cooling_rate, node_features):
        """
        Apply cross-attention between cooling rate and node features.
        
        Args:
            cooling_rate: Cooling rate tensor [batch_size]
            node_features: Node features [num_nodes, node_dim]
            
        Returns:
            Updated node features
        """
        # Get cooling rate embedding
        cooling_embed = self.base_embed(cooling_rate)
        cooling_embed = self.cooling_rate_projection(cooling_embed)
        
        # Repeat cooling rate embedding for each node
        batch_size = cooling_embed.size(0)
        nodes_per_batch = node_features.size(0) // batch_size
        cooling_embed = cooling_embed.unsqueeze(1).repeat(1, nodes_per_batch, 1)
        cooling_embed = cooling_embed.view(-1, cooling_embed.size(-1))
        
        # Apply cross-attention
        node_features_norm = self.norm1(node_features)
        attn_output, _ = self.attn(
            node_features_norm,  # Query
            cooling_embed.unsqueeze(1),  # Key
            cooling_embed.unsqueeze(1)   # Value
        )
        
        # Residual connection and layer norm
        node_features = node_features + attn_output.squeeze(1)
        node_features = node_features + self.ff(self.norm2(node_features))
        
        return node_features


# GaussianBasisEmbedding class from the original code for reference
class GaussianBasisEmbedding(nn.Module):
    """
    Embeds a scalar value in [0,1] using a Gaussian basis set followed by a dense layer.
    
    Args:
        num_basis: Number of Gaussian basis functions
        embedding_dim: Dimension of the final embedding vector
        min_sigma: Minimum sigma (width) for the Gaussian bases
        learn_means: Whether to learn the means of the Gaussian bases
        learn_sigmas: Whether to learn the sigmas of the Gaussian bases
    """
    def __init__(
        self, 
        num_basis=12, 
        embedding_dim=32, 
        min_sigma=0.1, 
        learn_means=False, 
        learn_sigmas=False,
        min_value=0,
        max_value=1,
    ):
        super().__init__()
        
        # Initialize means uniformly in [0,1]
        means = torch.linspace(min_value, max_value, num_basis)
        if learn_means:
            self.means = nn.Parameter(means)
        else:
            self.register_buffer('means', means)
            
        # Initialize sigmas with reasonable defaults
        sigmas = torch.ones_like(means) * max(min_sigma, 1.0 / (num_basis - 1))
        if learn_sigmas:
            self.sigmas = nn.Parameter(sigmas)
        else:
            self.register_buffer('sigmas', sigmas)
            
        # Two-layer neural network to produce the final embedding
        hidden_dim = max(embedding_dim * 2, num_basis)
        self.layer1 = nn.Linear(num_basis, hidden_dim)
        self.activation = nn.Softplus()
        self.layer2 = nn.Linear(hidden_dim, embedding_dim)
        
    def gaussian_basis(self, x):
        """Transform scalar input into Gaussian basis activations."""
        # Ensure x is 2D: [batch_size, 1]
        if x.dim() == 1:
            x = x.unsqueeze(1)
            
        # Compute Gaussian activations: exp(-(x-μ)²/(2σ²))
        # Shape: [batch_size, num_basis]
        x_expanded = x.expand(-1, self.means.shape[0])
        return torch.exp(-0.5 * ((x_expanded - self.means) / self.sigmas)**2)
        
    def forward(self, x):
        """
        Forward pass of the embedding module.
        
        Args:
            x: Input tensor of shape [batch_size] or [batch_size, 1]
               with values in range [0,1]
               
        Returns:
            Embedded representation of shape [batch_size, embedding_dim]
        """
        # Apply Gaussian basis transformation
        basis_activation = self.gaussian_basis(x)
        
        # Apply first layer and softplus activation
        hidden = self.activation(self.layer1(basis_activation))
        
        # Apply second linear layer to get the final embedding
        embedding = self.layer2(hidden)
        
        return embedding


# Complete NequIP model with improved cooling rate embedding
class NequIP_ImprovedCoolingRateEmbed(nn.Module):
    """NequIP model with improved cooling rate embedding techniques.
    
    This model extends the original NequIP architecture to incorporate cooling rate
    information using various advanced embedding and modulation techniques.
    
    Args:
        init_embed (function): Initial embedding function/class for nodes and edges.
        irreps_node_x (Irreps or str): Irreps of input node features.
        irreps_node_z (Irreps or str): Irreps of auxiliary node features (not updated throughout model).
        irreps_hidden (Irreps or str): Irreps of node features at hidden layers.
        irreps_edge (Irreps or str): Irreps of spherical_harmonics.
        irreps_out (Irreps or str): Irreps of output node features.
        num_convs (int): Number of interaction/conv layers. Must be more than 1.
        radial_neurons (list of ints): Number of neurons per layers in the MLP that learns from bond distances.
        num_neighbors (float): Typical or average node degree (used for normalization).
        embedding_type (str): Type of cooling rate embedding to use: "gaussian", "fourier", or "hierarchical".
        modulation_type (str): How to apply the cooling rate to the network: "add", "adaptive", 
                             "scale_shift", or "attention".
    """
    def __init__(self,
        init_embed,
        irreps_node_x  = '8x0e',
        irreps_node_z  = '8x0e',
        irreps_hidden  = '64x0e + 32x1e + 32x2e',
        irreps_edge    = '1x0e + 1x1e + 1x2e',
        irreps_out     = '1x1e',
        num_convs      = 3,
        radial_neurons = [16, 64],
        num_neighbors  = 12,
        embedding_type = "fourier",   # Options: "gaussian", "fourier", "hierarchical"
        modulation_type = "adaptive", # Options: "add", "adaptive", "scale_shift", "attention"
    ):
        super().__init__()
        self.init_embed     = init_embed
        self.irreps_node_x  = o3.Irreps(irreps_node_x)
        self.irreps_node_z  = o3.Irreps(irreps_node_z)
        self.irreps_hidden  = o3.Irreps(irreps_hidden)
        self.irreps_out     = o3.Irreps(irreps_out)
        self.irreps_edge    = o3.Irreps(irreps_edge)
        self.num_convs      = num_convs
        self.modulation_type = modulation_type

        act_scalars = {1: nn.functional.silu, -1: torch.tanh}
        act_gates   = {1: torch.sigmoid, -1: torch.tanh}

        irreps = self.irreps_node_x
        self.interactions = nn.ModuleList()
        for _ in range(num_convs):
            irreps_scalars = o3.Irreps([(m, ir) for m, ir in self.irreps_hidden if ir.l == 0 and tp_path_exists(irreps, self.irreps_edge, ir)])
            irreps_gated   = o3.Irreps([(m, ir) for m, ir in self.irreps_hidden if ir.l > 0  and tp_path_exists(irreps, self.irreps_edge, ir)])

            if irreps_gated.dim > 0:
                if tp_path_exists(irreps_node_z, self.irreps_edge, "0e"):
                    ir = "0e"
                elif tp_path_exists(irreps_node_z, self.irreps_edge, "0o"):
                    ir = "0o"
                else:
                    raise ValueError(f"irreps={irreps} times irreps_edge={self.irreps_edge} is unable to produce gates needed for irreps_gated={irreps_gated}.")
            else:
                ir = None
            irreps_gates = o3.Irreps([(mul, ir) for mul, _ in irreps_gated]).simplify()

            gate = Gate(
                irreps_scalars, [act_scalars[ir.p] for _, ir in irreps_scalars],  # scalar
                irreps_gates,   [act_gates[ir.p]   for _, ir in irreps_gates],  # gates (scalars)
                irreps_gated  # gated tensors
            )

            conv = Interaction(
                irreps_in      = irreps,
                irreps_node    = self.irreps_node_z,
                irreps_edge    = self.irreps_edge,
                irreps_out     = gate.irreps_in,
                radial_neurons = radial_neurons,
                num_neighbors  = num_neighbors,
            )
            irreps = gate.irreps_out
            self.interactions.append(Compose(conv, gate))

        self.out = o3.FullyConnectedTensorProduct(
            irreps_in1 = irreps,
            irreps_in2 = self.irreps_node_z,
            irreps_out = self.irreps_out,
        )

        # Get embedding dimension from final irreps
        size_embed = int(str(irreps).split("x")[0])
        
        # Create cooling rate embedding based on selected type
        if embedding_type == "gaussian":
            self.cooling_embed = GaussianBasisEmbedding(
                embedding_dim=size_embed,
                num_basis=6, 
                min_value=-4,  # log10(0.0001)
                max_value=4,   # log10(10000)
                min_sigma= 1.0  # Wider Gaussian bases
            )
        elif embedding_type == "fourier":
            self.cooling_embed = FourierFeatureEmbedding(
                embedding_dim=size_embed,
                scale=10.0,
                min_value=-4,
                max_value=4
            )
        elif embedding_type == "hierarchical":
            self.cooling_embed = HierarchicalCoolingRateEmbedding(
                embedding_dim=size_embed,
                num_levels=3,
                min_value=-4,
                max_value=4
            )
        else:
            raise ValueError(f"Unknown embedding_type: {embedding_type}")
        
        # Create modulation based on selected type
        if modulation_type == "add":
            # Simple addition (like in original)
            self.cooling_projection = nn.Sequential(
                nn.Linear(size_embed, size_embed),
                nn.SiLU(),
                nn.Linear(size_embed, irreps.dim)
            )
        elif modulation_type == "adaptive":
            self.cooling_modulation = AdaptiveLayerModulation(
                self.cooling_embed,
                num_layers=num_convs,
                feature_dim=irreps.dim
            )
        elif modulation_type == "scale_shift":
            self.cooling_modulation = ScaleAndShiftModulation(
                self.cooling_embed,
                feature_dim=irreps.dim
            )
        elif modulation_type == "attention":
            self.cooling_modulation = CrossAttentionEmbedding(
                self.cooling_embed,
                node_dim=irreps.dim
            )
        else:
            raise ValueError(f"Unknown modulation_type: {modulation_type}")

    def forward(self, data, cooling_rate):
        """
        Forward pass of the NequIP model.
        
        Args:
            data: PyG data object containing the graph
            cooling_rate: Tensor of cooling rates (should be log-transformed already)
                         Shape: [batch_size]
        
        Returns:
            Predicted displacement vectors
        """
        # Embedding
        data = self.init_embed(data)
        edge_index, edge_attr = data.edge_index, data.edge_attr
        h_node_x, h_node_z, h_edge = data.h_node_x, data.h_node_z, data.h_edge
        
        # Get batch assignment (which graph each node belongs to)
        batch = getattr(data, 'batch', torch.zeros(h_node_x.size(0), device=h_node_x.device, dtype=torch.long))
        
        # Compute spherical harmonics for edge attributes
        edge_sh = o3.spherical_harmonics(self.irreps_edge, edge_attr, normalize=True, normalization='component')
        
        # Apply different modulation approaches based on selected type
        if self.modulation_type == "add":
            # Original approach (embed + project + add)
            cooling_embed = self.cooling_embed(cooling_rate)
            cooling_features = self.cooling_projection(cooling_embed)
            cooling_features = cooling_features[batch]  # Expand to all nodes
            
            # Add in each layer
            for layer in self.interactions:
                h_node_x = layer(h_node_x, h_node_z, edge_index, edge_sh, h_edge)
                h_node_x = h_node_x + cooling_features
            
        elif self.modulation_type == "adaptive":
            # Get layer-specific modulations
            layer_modulations = self.cooling_modulation(cooling_rate)
            
            # Apply different modulation at each layer
            for i, layer in enumerate(self.interactions):
                h_node_x = layer(h_node_x, h_node_z, edge_index, edge_sh, h_edge)
                layer_mod = layer_modulations[i][batch]  # Expand to all nodes
                h_node_x = h_node_x + layer_mod
            
        elif self.modulation_type == "scale_shift":
            # Apply scale and shift modulation at each layer
            for layer in self.interactions:
                h_node_x = layer(h_node_x, h_node_z, edge_index, edge_sh, h_edge)
                h_node_x = self.cooling_modulation(cooling_rate, h_node_x)
            
        elif self.modulation_type == "attention":
            # Apply attention-based modulation at each layer
            for layer in self.interactions:
                h_node_x = layer(h_node_x, h_node_z, edge_index, edge_sh, h_edge)
                h_node_x = self.cooling_modulation(cooling_rate, h_node_x)
        
        # Final output layer
        return self.out(h_node_x, h_node_z)