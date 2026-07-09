from dmol.diff_mfld.bundle.vector_bundle import ScalarBundle
from dmol.diff_mfld.field import Field
from dmol.diff_mfld.field.field_types import LambdaField
from dmol.diff_mfld.field.util import coord_repr
from dmol.diff_mfld.mfld import Manifold


class TestSpecs:
    def test_top_level(self):
        M = Manifold[2]

        scalar_lambda_field = LambdaField[ScalarBundle[M]](lambda x, y: coord_repr(x * y))  # type: ignore
        assert scalar_lambda_field._top_level_type is Field  # type: ignore
