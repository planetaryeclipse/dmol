from dmol.diff_mfld.connection.connection import Connection
import numpy as np
import torch

from enum import Enum
from typing import Union, Callable, Tuple

from scipy.integrate import solve_ivp

from dmol.diff_mfld.util import specs_match
from dmol.diff_mfld.mfld import Manifold, Point
from dmol.diff_mfld.connection.connection import BundleCurve
from dmol.diff_mfld.geometry.vector_bundle import VectorBundle
from dmol.diff_mfld.geometry.riemannian import EuclideanMetricField
from dmol.diff_mfld.geometry.tensor import Vec


def _exp_map_ivp_fn(
    t, y: np.ndarray, n: int, coeffs: Callable[[np.ndarray], np.ndarray]
) -> np.ndarray:
    p, v = y[:n], y[n:]

    print(f"p: {p}")
    print(f"v: {v}")

    conn_coeffs = coeffs(p)

    exit()

    dot_p = v
    dot_v = -np.einsum("kij,i,j->k", conn_coeffs, v, v)
    return np.concat((dot_p, dot_v))


def ivp_exp_map[T: VectorBundle](
    p: Union[Point[T], torch.Tensor],
    v: Vec[T],
    coeffs: Callable[[Union[Point[T], torch.Tensor]], torch.Tensor],
) -> Tuple[Point[T], BundleCurve[T]]:
    p = Point[v.bundle](p)

    print(coeffs)

    # NOTE: numpy is needed for use in scipy methods
    coeffs_np = lambda p: coeffs(torch.from_numpy(p)).detach().numpy()

    print(f"p: {p}")
    print(f"p.p: {p.p}")

    result = solve_ivp(
        _exp_map_ivp_fn,
        [0.0, 1.0],
        np.concat((p.p.detach().numpy(), v.components.detach().numpy())),
        method="Radau",
        args=(p.manifold.dim, coeffs_np),
    )

    t = result.t.T
    p = result.y[:, : p.manifold.dim].T
    v = result.y[:, p.manifold.dim :].T

    return (Point[p.manifold](p[-1, :]), BundleCurve[v.bundle](t, p, (v,)))



# def _parallel_transp_ivp_fn(
#     t,
#     u: np.ndarray,
#     coeffs: Callable[[np.ndarray], np.ndarray],
#     curve: Callable[[float], Tuple[np.ndarray, np.ndarray]],
# ) -> np.ndarray:
#     p, v = curve(t)  # time-parameterized curve
#     conn_coeffs = coeffs(p)

#     dot_u = -np.einsum("i,j,kij->k", v, u, conn_coeffs)
#     return dot_u


class ExpMapMethod(Enum):
    IVP = (ivp_exp_map,)  # wrap in tuple to force execution through __call__

    def __call__[T: VectorBundle](
        self, p: Union[Point[T], torch.Tensor], v: Vec[T], conn: Connection[T]
    ):
        print(f"vec bundle: {v.bundle}")
        print(f"conn bundle: {conn.bundle}")

        p = Point[v.bundle](p)

        print()
        print(v.bundle.__dict__)
        print()

        if not specs_match(v.bundle, conn.bundle):
            raise TypeError(
                f"bundles of provided vector {v.bundle.__name__} and connection {conn.bundle.__name__} do not match"
            )

        (exp_map_handler,) = self.value
        coeffs = conn.coeffs

        return exp_map_handler(p, v, coeffs)


M2 = Manifold[2]
p = torch.tensor([2.0, 3.0])
v = Vec[M2](torch.tensor([4.0, 5.0]))

euclid = EuclideanMetricField[M2]()

print()
print(f"euclid: {euclid}")
print()

flat_conn = euclid.levi_civita()

flat_conn.partials(p)

exit()

ExpMapMethod.IVP(p, v, flat_conn)
