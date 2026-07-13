# ensure correct order is loaded to avoid cyclic dependencies

from .bundle.vector_bundle import VectorBundle
from .bundle.tensor import Tensor
from .field import Field
from .connection import Connection

# ensure torch and numpy types can directly be exchanged
import torch

torch.set_default_dtype(torch.float64)
torch.set_default_device("cpu")
