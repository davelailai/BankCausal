from functools import partial
from typing import Optional, Union, cast

import torch
from einops import rearrange
from torch import Tensor, nn



from copy import deepcopy
from typing import Callable, Optional, Union

import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.modules.transformer import _get_activation_fn


import torch
from einops import einsum, rearrange
import math

import torch
from einops import einsum, rearrange
from torch import Tensor, nn
from mmpretrain.registry import MODELS


class MultiExpertLayer(nn.Module):
    """A more efficient alternative to creating 'n' separate expert layers (likely
    from 'nn.Linear' modules).  Instead, we create a single set of batched weights
    and biases, and apply all 'experts' in parallel.

    Args:
        embed_dim (int): embedding dimension (d)
        num_experts (int): number of experts (n)
        bias (bool): whether to include a bias term. Default: True
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_experts: int,
        bias: bool = True,
        device: Optional[Union[torch.device, str]] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_experts = num_experts

        self.weight = nn.Parameter(
            torch.empty(
                (num_experts, in_features, out_features), device=device, dtype=dtype
            )
        )
        bias_param: Optional[nn.Parameter] = None
        if bias:
            bias_param = nn.Parameter(
                torch.empty((num_experts, out_features), device=device, dtype=dtype)
            )
        # Include type annotation for mypy :D
        self.bias: Optional[nn.Parameter]
        self.register_parameter("bias", bias_param)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        # NOTE: Mostly copy-pasta from 'nn.Linear.reset_parameters'
        #
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: Tensor) -> Tensor:
        if x.size(-1) != self.in_features:
            raise ValueError(
                f"Expected input with embed_dim={self.in_features} (dim=-1), but "
                f"found {x.size(-1)}"
            )
        elif x.size(1) != self.num_experts:
            raise ValueError(
                f"Expected input with num_experts={self.num_experts} (dim=1), but "
                f"found {x.size(1)}"
            )

        # NOTE: 'd1' and 'd2' are both equal to 'embed_dim'. But for 'einsum' to
        # work correctly, we have to give them different names.
        x = einsum(x, self.weight, "b n ... d1, n d1 d2 -> b n ... d2")

        if self.bias is not None:
            # NOTE: When used with 'SoftMoE' the inputs to 'MultiExpertLayer' will
            # always be 4-dimensional.  But it's easy enough to generalize for 3D
            # inputs as well, so I decided to include that here.
            if x.ndim == 3:
                bias = rearrange(self.bias, "n d -> () n d")
            elif x.ndim == 4:
                bias = rearrange(self.bias, "n d -> () n () d")
            else:
                raise ValueError(
                    f"Expected input to have 3 or 4 dimensions, but got {x.ndim}"
                )
            x = x + bias

        return x

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"num_experts={self.num_experts}, bias={self.bias is not None}"
        )
    
class GroupedSoftMoE(nn.Module):
    """A grouped Soft-MoE module for processing multiple classes independently.

    Args:
        in_features (int): Input feature dimension (d).
        out_features (int): Output feature dimension (d_out).
        num_experts (int): Number of experts (n).
        slots_per_expert (int): Number of slots per expert (p).
        num_classes (int): Number of classes (L).
        bias (bool): Whether to include a bias term. Default: True.
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_experts: int,
        slots_per_expert: int,
        num_classes: int,
        bias: bool = True,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.in_features = in_features
        self.out_features = out_features
        self.num_experts = num_experts
        self.slots_per_expert = slots_per_expert

        # Create a separate Soft-MoE for each class (L)
        self.classwise_smoe = nn.ModuleList([
            SoftMoE(
                in_features=in_features,
                out_features=out_features,
                num_experts=num_experts,
                slots_per_expert=slots_per_expert,
                bias=bias,
                device=device,
                dtype=dtype
            )
            for _ in range(num_classes)
        ])

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass for the grouped Soft-MoE.
        
        Args:
            x (Tensor): Input tensor of shape (b, L, m, d),
                        where L is the number of classes, m is sequence length,
                        and d is the embedding dimension.
        
        Returns:
            Tensor: Output tensor of shape (b, L, m, d_out).
        """
        if x.ndim != 4:
            raise ValueError(
                f"Expected input tensor with 4 dimensions (b, L, m, d), got {x.ndim} dimensions."
            )
        batch_size, num_classes, seq_len, embed_dim = x.shape
        if num_classes != self.num_classes:
            raise ValueError(
                f"Input num_classes={num_classes} does not match the defined number of classes={self.num_classes}."
            )
        if embed_dim != self.in_features:
            raise ValueError(
                f"Input embedding dim={embed_dim} does not match the expected in_features={self.in_features}."
            )

        # Process each class independently with its corresponding Soft-MoE
        outputs = []
        for l in range(self.num_classes):
            class_input = x[:, l, :, :]  # Shape: (b, m, d)
            class_output = self.classwise_smoe[l](class_input)  # Shape: (b, m, d_out)
            outputs.append(class_output)

        # Combine outputs along the class (L) dimension
        return torch.stack(outputs, dim=1)  # Shape: (b, L, m, d_out)


class SoftMoE(nn.Module):
    """A PyTorch module for Soft-MoE, as described in the paper:
        "From Sparse to Soft Mixtures of Experts"
        https://arxiv.org/pdf/2308.00951.pdf

    einstein notation:
    - b: batch size
    - m: input sequence length
    - d: embedding dimension
    - n: num experts
    - p: num slots per expert
    - (n * p): total number of slots

    Args:
        embed_dim (int): embedding dimension (d)
        num_experts (int): number of experts (n)
        slots_per_expert (int): number of slots per expert (p)
        bias (bool): whether to include a bias term. Default: True.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_experts: int,
        slots_per_expert: int,
        bias: bool = True,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
        # normalize: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_experts = num_experts
        self.slots_per_expert = slots_per_expert
        self.bias = bias
        # self.normalize = normalize

        # if self.normalize:
        #     self.scale = nn.Parameter(torch.ones(1))

        self.phi = nn.Parameter(
            torch.empty(
                (in_features, num_experts, slots_per_expert),
                device=device,
                dtype=dtype,
            )
        )
        self.experts = MultiExpertLayer(
            in_features=in_features,
            out_features=out_features,
            num_experts=num_experts,
            bias=bias,
            device=device,
            dtype=dtype,
        )
          
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # NOTE: Copy weight initialization from 'nn.Linear.reset_parameters'
        # TODO: Check for initialization strategy from the paper
        nn.init.kaiming_uniform_(self.phi, a=math.sqrt(5))
        
    def multi_softmax(self, x: torch.Tensor, dim: int) -> torch.Tensor:
        """
        Compute the softmax along the specified dimensions.
        This function adds the option to specify multiple dimensions

        Args:
            x (torch.Tensor): Input tensor.
            dims (int or tuple[int]): The dimension or list of dimensions along which the softmax probabilities are computed.

        Returns:
            torch.Tensor: Output tensor containing softmax probabilities along the specified dimensions.
        """
        max_vals = torch.amax(x, dim=dim, keepdim=True)
        e_x = torch.exp(x - max_vals)
        sum_exp = e_x.sum(dim=dim, keepdim=True)
        return e_x / sum_exp

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass for the Soft-MoE layer, as described in:
            https://arxiv.org/pdf/2308.00951.pdf
        See: equations (1-3), algorithm 1, and figure 2

        einstein notation:
        - b: batch size
        - m: input sequence length
        - d: embedding dimension
        - n: num experts
        - p: num slots per expert
        - (n * p): total number of slots

        Args:
            x (Tensor): input tensor of shape (b, m, d)

        Returns:
            Tensor: output tensor of shape (b, m, d)
        """
        if x.size(-1) != self.in_features:
            raise ValueError(
                f"Expected x.size(-1)={x.size(-1)} to match embed_dim={self.in_features}, "
                f"but got {x.size(-1)}."
            )
        elif x.ndim != 3:
            raise ValueError(f"Expected input to have 3 dimensions, but got {x.ndim}.")
        # phi = self.phi

        # Normalize input and phi
        # if self.normalize:
        #     x = F.normalize(x, dim=2)  # [b, m, d]
            # phi = self.scale * F.normalize(self.phi, dim=0)  # [d, n, p]

        logits = einsum(x, self.phi, "b m d, d n p -> b m n p")
        d = self.multi_softmax(logits, dim=1)
        c = self.multi_softmax(logits, dim=(2, 3))
         # Compute input slots as weighted average of input tokens using dispatch weights
        x = torch.einsum("bmd,bmnp->bnpd", x, d)
        x = self.experts(x)
        
        x = einsum(x, c, "b n p d, b m n p -> b m d")

        # Apply expert to corresponding slots
        # ys = torch.stack(
        #     [f_i(xs[:, i, :, :]) for i, f_i in enumerate(self.experts)], dim=1
        # )

        # # Compute output tokens as weighted average of output slots using combine weights
        # y = torch.einsum("bnpd,bmnp->bmd", ys, c)

        # return y


        # dispatch_weights = logits.softmax(dim=1)  # denoted 'D' in the paper
        # # NOTE: The 'torch.softmax' function does not support multiple values for the
        # # 'dim' argument (unlike jax), so we are forced to flatten the last two dimensions.
        # # Then, we rearrange the Tensor into its original shape.
        # combine_weights = rearrange(
        #     logits.flatten(start_dim=2).softmax(dim=-1),
        #     "b m (n p) -> b m n p",
        #     n=self.num_experts,
        # )

        # # NOTE: To save memory, I don't rename the intermediate tensors Y, Ys, Xs.
        # # Instead, I just overwrite the 'x' variable.  The names from the paper are
        # # included in a comment for each line below.
        # x = einsum(x, dispatch_weights, "b m d, b m n p -> b n p d")  # Xs
        # x = self.experts(x)  # Ys
        # x = einsum(x, combine_weights, "b n p d, b m n p -> b m d")  # Y

        return x

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"num_experts={self.num_experts}, slots_per_expert={self.slots_per_expert}, "
            f"bias={self.bias}"
        )


import torch
from torch import nn, Tensor
from typing import Optional, Union

class GroupedLinear(nn.Module):
    """A grouped module with separate linear layers for processing multiple classes independently.

    Args:
        in_features (int): Input feature dimension (d).
        out_features (int): Output feature dimension (d_out).
        num_classes (int): Number of classes (L).
        bias (bool): Whether to include a bias term. Default: True.
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_classes: int,
        bias: bool = True,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.in_features = in_features
        self.out_features = out_features

        # Create a separate linear layer for each class (L)
        self.classwise_linear = nn.ModuleList([
            nn.Linear(
                in_features=in_features,
                out_features=out_features,
                bias=bias,
                device=device,
                dtype=dtype
            )
            for _ in range(num_classes)
        ])

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass for the grouped module with linear layers.
        
        Args:
            x (Tensor): Input tensor of shape (b, L, m, d),
                        where L is the number of classes, m is sequence length,
                        and d is the embedding dimension.
        
        Returns:
            Tensor: Output tensor of shape (b, L, m, d_out).
        """
        if x.ndim != 3:
            raise ValueError(
                f"Expected input tensor with 4 dimensions (b, L, m, d), got {x.ndim} dimensions."
            )
        batch_size, num_classes, embed_dim = x.shape
        if num_classes != self.num_classes:
            raise ValueError(
                f"Input num_classes={num_classes} does not match the defined number of classes={self.num_classes}."
            )
        if embed_dim != self.in_features:
            raise ValueError(
                f"Input embedding dim={embed_dim} does not match the expected in_features={self.in_features}."
            )

        # Process each class independently with its corresponding linear layer
        outputs = []
        for l in range(self.num_classes):
            class_input = x[:, l, :]  # Shape: (b, m, d)
            class_input = class_input.view(-1, self.in_features)  # Reshape to (b * m, d)
            class_output = self.classwise_linear[l](class_input)  # Shape: (b * m, d_out)
            class_output = class_output.view(batch_size, self.out_features)  # Reshape back to (b, m, d_out)
            outputs.append(class_output)

        # Combine outputs along the class (L) dimension
        return torch.stack(outputs, dim=1)  # Shape: (b, L, m, d_out)