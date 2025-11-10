import torch
import torch.nn as nn

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

# ======
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
# =====

# Original NequIP
class NequIP(nn.Module):
    """NequIP model from https://arxiv.org/pdf/2101.03164.pdf.

    Args:
        init_embed (function): Initial embedding function/class for nodes and edges.
        irreps_node_x (Irreps or str): Irreps of input node features.
        irreps_node_z (Irreps or str): Irreps of auxiliary node features (not updated throughout model).
        irreps_hidden (Irreps or str): Irreps of node features at hidden layers.
        irreps_edge (Irreps or str): Irreps of spherical_harmonics.
        irreps_out (Irreps or str): Irreps of output node features.
        num_convs (int): Number of interaction/conv layers. Must be more than 1.
        radial_neurons (list of ints): Number of neurons per layers in the MLP that learns from bond distances.
            For first and hidden layers, not the output layer.
        max_radius (float): Cutoff radius used during graph construction.
        num_neighbors (float): Typical or average node degree (used for normalization).
    
    Notes:
        The `init_embed` function/class must take a PyG graph object `data` as input and output the same object
        with the additional fields `h_node_x`, `h_node_z`, and `h_edge` that correspond to the node, auxilliary node,
        and edge embeddings.
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
    ):
        super().__init__()
        self.init_embed     = init_embed
        self.irreps_node_x  = o3.Irreps(irreps_node_x)
        self.irreps_node_z  = o3.Irreps(irreps_node_z)
        self.irreps_hidden  = o3.Irreps(irreps_hidden)
        self.irreps_out     = o3.Irreps(irreps_out)
        self.irreps_edge    = o3.Irreps(irreps_edge)
        self.num_convs      = num_convs

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

    def forward(self, data):
        # Embedding
        data = self.init_embed(data)
        edge_index, edge_attr = data.edge_index, data.edge_attr
        h_node_x, h_node_z, h_edge = data.h_node_x, data.h_node_z, data.h_edge

        # Graph convolutions
        edge_sh = o3.spherical_harmonics(self.irreps_edge, edge_attr, normalize=True, normalization='component')
        for layer in self.interactions:
            h_node_x = layer(h_node_x, h_node_z, edge_index, edge_sh, h_edge)

        # Final output layer
        return self.out(h_node_x, h_node_z)

# TimeEmbedded NequIP
class NequIP_TimeEmbed(nn.Module):
    """NequIP model from https://arxiv.org/pdf/2101.03164.pdf.

    Args:
        init_embed (function): Initial embedding function/class for nodes and edges.
        irreps_node_x (Irreps or str): Irreps of input node features.
        irreps_node_z (Irreps or str): Irreps of auxiliary node features (not updated throughout model).
        irreps_hidden (Irreps or str): Irreps of node features at hidden layers.
        irreps_edge (Irreps or str): Irreps of spherical_harmonics.
        irreps_out (Irreps or str): Irreps of output node features.
        num_convs (int): Number of interaction/conv layers. Must be more than 1.
        radial_neurons (list of ints): Number of neurons per layers in the MLP that learns from bond distances.
            For first and hidden layers, not the output layer.
        max_radius (float): Cutoff radius used during graph construction.
        num_neighbors (float): Typical or average node degree (used for normalization).
    
    Notes:
        The `init_embed` function/class must take a PyG graph object `data` as input and output the same object
        with the additional fields `h_node_x`, `h_node_z`, and `h_edge` that correspond to the node, auxilliary node,
        and edge embeddings.
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
    ):
        super().__init__()
        self.init_embed     = init_embed
        self.irreps_node_x  = o3.Irreps(irreps_node_x)
        self.irreps_node_z  = o3.Irreps(irreps_node_z)
        self.irreps_hidden  = o3.Irreps(irreps_hidden)
        self.irreps_out     = o3.Irreps(irreps_out)
        self.irreps_edge    = o3.Irreps(irreps_edge)
        self.num_convs      = num_convs

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

        # size_embed = int(irreps.split("x")[0])
        size_embed = int(str(irreps).split("x")[0]) # change for String
        self.t_embed = GaussianBasisEmbedding(embedding_dim=size_embed)
        # self.y_embed = GaussianBasisEmbedding(embedding_dim=size_embed, min_value=-3, max_value=3) # for cooling rate

        #===new===
        # Add projection layer for time embedding
        t_embed_dim = self.t_embed.layer2.out_features  # Get embedding dimension
        self.t_projection = nn.Linear(t_embed_dim, irreps.dim)
        #===end===

    def forward(self, data, t):
        # Embedding
        data = self.init_embed(data)
        edge_index, edge_attr = data.edge_index, data.edge_attr
        h_node_x, h_node_z, h_edge = data.h_node_x, data.h_node_z, data.h_edge

        # time embedding
        h_node_t = self.t_embed(t)
        #===new===
        # Expand to match number of nodes
        h_node_t = h_node_t.expand(h_node_x.shape[0], -1)  # [num_nodes, embedding_dim]
        # Add a projection layer in __init__:
        # self.t_projection = nn.Linear(embedding_dim, irreps.dim)
        h_node_t = self.t_projection(h_node_t)  # [num_nodes, irreps.dim]
        #===end===
        
    
        # h_node_y = self.y_embed(data.y) # for cooling rate
        
        # Graph convolutions
        edge_sh = o3.spherical_harmonics(self.irreps_edge, edge_attr, normalize=True, normalization='component')
        for layer in self.interactions:
            h_node_x = layer(h_node_x, h_node_z, edge_index, edge_sh, h_edge)
            h_node_x = h_node_x + h_node_t
            # h_node_x = h_node_x + h_node_y # for cooling rate

        # Final output layer
        return self.out(h_node_x, h_node_z)


# Cooling_rate Embedded NequIP
class NequIP_CoolingRateEmbed(nn.Module):
    """NequIP model from https://arxiv.org/pdf/2101.03164.pdf.

    Args:
        init_embed (function): Initial embedding function/class for nodes and edges.
        irreps_node_x (Irreps or str): Irreps of input node features.
        irreps_node_z (Irreps or str): Irreps of auxiliary node features (not updated throughout model).
        irreps_hidden (Irreps or str): Irreps of node features at hidden layers.
        irreps_edge (Irreps or str): Irreps of spherical_harmonics.
        irreps_out (Irreps or str): Irreps of output node features.
        num_convs (int): Number of interaction/conv layers. Must be more than 1.
        radial_neurons (list of ints): Number of neurons per layers in the MLP that learns from bond distances.
            For first and hidden layers, not the output layer.
        max_radius (float): Cutoff radius used during graph construction.
        num_neighbors (float): Typical or average node degree (used for normalization).
    
    Notes:
        The `init_embed` function/class must take a PyG graph object `data` as input and output the same object
        with the additional fields `h_node_x`, `h_node_z`, and `h_edge` that correspond to the node, auxilliary node,
        and edge embeddings.
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
    ):
        super().__init__()
        self.init_embed     = init_embed
        self.irreps_node_x  = o3.Irreps(irreps_node_x)
        self.irreps_node_z  = o3.Irreps(irreps_node_z)
        self.irreps_hidden  = o3.Irreps(irreps_hidden)
        self.irreps_out     = o3.Irreps(irreps_out)
        self.irreps_edge    = o3.Irreps(irreps_edge)
        self.num_convs      = num_convs

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

        # size_embed = int(irreps.split("x")[0])
        size_embed = int(str(irreps).split("x")[0]) # change for String
        # self.t_embed = GaussianBasisEmbedding(embedding_dim=size_embed)
        self.y_embed = GaussianBasisEmbedding(
                            embedding_dim=size_embed,
                            num_basis=9, 
                            min_value=-4, 
                            max_value=4,
                            min_sigma=0.6  # Changed from default 0.1
                        )

        #===new===
        # Add projection layer for time embedding
        y_embed_dim = self.y_embed.layer2.out_features  # Get embedding dimension
        self.y_projection = nn.Sequential(
                                nn.Linear(y_embed_dim, y_embed_dim), 
                                nn.SiLU(),  # SiLU activation (can replace with nn.Tanh() if preferred)
                                nn.Linear(y_embed_dim, irreps.dim) 
                            )
        #===end===  

    def forward(self, data, y):
        # Embedding
        data = self.init_embed(data)
        edge_index, edge_attr = data.edge_index, data.edge_attr
        h_node_x, h_node_z, h_edge = data.h_node_x, data.h_node_z, data.h_edge
        
        # Get batch assignment (which graph each node belongs to)
        batch = getattr(data, 'batch', torch.zeros(h_node_x.size(0), device=h_node_x.device, dtype=torch.long))
        
        # Embed cooling rate
        h_node_y = self.y_embed(y)  # [batch_size, embed_dim]
        
        # Create a node-level embedding through indexing
        h_node_y = h_node_y[batch]  # This indexes the appropriate cooling rate for each node
        
        # Project to match dimension for network
        h_node_y = self.y_projection(h_node_y)
        
        # Graph convolutions
        edge_sh = o3.spherical_harmonics(self.irreps_edge, edge_attr, normalize=True, normalization='component')
        for layer in self.interactions:
            h_node_x = layer(h_node_x, h_node_z, edge_index, edge_sh, h_edge)
            h_node_x = h_node_x + h_node_y  # Add cooling rate info at each layer
            
        # Final output layer
        return self.out(h_node_x, h_node_z) 