from dataclasses import dataclass

import torch

from dmol.diff_mfld.bundle.tensor import Scalar
from dmol.diff_mfld.mfld import Manifold, Point


@dataclass
class ConstrResult[M: Manifold]:
    success: bool
    num_iters: int
    p: Point[M]
    f: Scalar[M]
    ineqs: list[Scalar[M]]
    eqs: list[Scalar[M]]

    # not all algorithms provide these multipliers
    ineq_mults: torch.Tensor | None = None
    eq_mults: torch.Tensor | None = None

    f_hist: torch.Tensor | None = None
    ineqs_hist: torch.Tensor | None = None
    eqs_hist: torch.Tensor | None = None
    p_hist: torch.Tensor | None = None

    def add_hist(
        self,
        f_hist: list[float],
        ineqs_hist: list[torch.Tensor],
        eqs_hist: list[torch.Tensor],
        p_hist: list[torch.Tensor],
    ):
        num_samples = len(f_hist)
        num_ineqs = len(ineqs_hist[0])
        num_eqs = len(eqs_hist[0])

        f_hist_tens = torch.zeros((num_samples,))
        ineqs_hist_tens = torch.zeros((num_ineqs, num_samples))
        eqs_hist_tens = torch.zeros((num_eqs, num_samples))
        p_hist_tens = torch.zeros((self.p.manifold.dim, num_samples))  # type: ignore

        for i in range(num_samples):
            f_hist_tens[i] = f_hist[i]
            ineqs_hist_tens[:, i] = ineqs_hist[i]
            eqs_hist_tens[:, i] = eqs_hist[i]
            p_hist_tens[:, i] = p_hist[i]

        self.f_hist = f_hist_tens
        self.ineqs_hist = ineqs_hist_tens
        self.eqs_hist = eqs_hist_tens
        self.p_hist = p_hist_tens
