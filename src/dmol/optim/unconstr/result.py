from dataclasses import dataclass

import torch

from dmol.diff_mfld.bundle.tensor import Scalar
from dmol.diff_mfld.mfld import Manifold, Point


@dataclass
class UnconstrResult[M: Manifold]:
    success: bool
    num_iters: int
    p: Point[M]
    f: Scalar[M]
    f_hist: torch.Tensor | None = None
    p_hist: torch.Tensor | None = None

    def add_hist(self, f_hist: list[float], p_hist: list[torch.Tensor]):
        num_samples = len(f_hist)  # type: ignore
        f_hist_tens = torch.zeros((num_samples,))
        p_hist_tens = torch.zeros((self.p.manifold.dim, num_samples))  # type: ignore
        for i in range(num_samples):
            f_hist_tens[i] = f_hist[i]  # type: ignore
            p_hist_tens[i, :] = p_hist[i]  # type: ignore

        self.f_hist = f_hist_tens
        self.p_hist = p_hist_tens
