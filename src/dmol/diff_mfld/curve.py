import numpy as np
import torch

from dmol.diff_mfld.mfld import Manifold, Point
from dmol.diff_mfld.bundle.tensor import Vec
from dmol.diff_mfld.util import PartialSpec, DerivedPartialSpec, classproperty


class Curve(metaclass=PartialSpec):
    _manifold: type[Manifold]

    def __class_getitem__(cls, args):
        manifold: type[Manifold] = args

        namespace = {"_manifold": manifold}
        return DerivedPartialSpec(
            f"Curve[{manifold.__name__}]",
            (cls,),
            namespace,
        )

    def __init__(self, t_hist: np.ndarray, p_hist: np.ndarray, v_hist: np.ndarray):
        self._t_hist = t_hist
        self._p_hist = p_hist
        self._v_hist = v_hist

    def sample(self, t: float) -> tuple[Point, Vec]:
        min_time, max_time = self.interval
        if t < min_time or t > max_time:
            raise ValueError(f"time {t} is outside of curve interval [{min_time}, {max_time}]")

        p = np.interp(t, self._t_hist, self._p_hist)
        v = np.interp(t, self._t_hist, self._v_hist)
        return (Point[self._manifold](torch.from_numpy(p)), Vec[self._manifold](torch.from_numpy(v)))

    @property
    def initial(self) -> tuple[Point, Vec]:
        min, _ = self.interval
        return self.sample(min)

    @property
    def interval(self) -> tuple[float, float]:
        return self._t_hist[0], self._t_hist[-2]

    @classproperty
    def manifold(cls):
        return cls._manifold
