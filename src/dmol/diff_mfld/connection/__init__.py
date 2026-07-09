from .base import Connection
from .covar_diff import _connection_covar, _connection_total_covar

Connection.covar = _connection_covar
Connection.total_covar = _connection_total_covar

__all__ = ["Connection"]
