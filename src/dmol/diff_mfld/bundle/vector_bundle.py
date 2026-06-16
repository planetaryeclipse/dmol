from abc import abstractmethod
from math import prod
from typing import Tuple, Dict, Optional

from dmol.diff_mfld.mfld import Manifold
from dmol.diff_mfld.util import classproperty, PartialSpec


class BundleIndices:
    @abstractmethod
    @classproperty
    def bundles(cls) -> Tuple[VectorBundle, ...]:
        raise NotImplementedError()

    @abstractmethod
    @classproperty
    def bundle_indices(cls) -> Dict[VectorBundle, Tuple[int, ...]]:
        raise NotImplementedError()


class VectorBundle(Manifold, BundleIndices):
    _rank: int = None
    _base: Optional[type[Manifold]] = None

    def __class_getitem__(cls, args):
        rank: int
        base: Optional[type[Manifold]] = None

        if type(args) is int:
            rank, base = args, None
        elif len(args) == 2:
            rank, base = args

            if base is None:
                raise TypeError("explicitly provided base type is unable to be none")
            elif base.incomplete:
                raise TypeError("base manifold type not fully specified")

        namespace = {
            "_dim": rank,
            "_rank": rank,
            "_base": base,
        }
        if base is None:
            namespace.update({"__class_getitem__": cls._partial_spec_gen_base})

        gen_cls = PartialSpec(
            f"VectorBundle[{rank, base}]",
            (VectorBundle,),
            namespace,
            creating_derived=True,
        )
        return gen_cls

    def _partial_spec_gen_base(cls, args):
        base: type[Manifold] = args

        namespace = {"_dim": cls._rank, "_rank": cls._rank, "_base": base}
        if isinstance(base, VectorBundle):
            if base.incomplete:
                namespace.update(
                    {"__class_getitem__": VectorBundle._partial_spec_mfld_pass_to_base}
                )
        elif base.incomplete:
            # print(base)
            # print(base._incomplete)
            raise TypeError("base manifold type not fully specified")

        return PartialSpec(
            f"VectorBundle[{cls._rank}, {base.__name__}]",
            (cls,),  # prevent generating chain of random unspecialized classes
            namespace,
            creating_derived=True,
            top_level_type=cls._top_level_type,
        )

    def _partial_spec_mfld_pass_to_base(cls, args):
        underlying_base: type[Manifold] = args

        upd_base = cls._base[underlying_base]
        namespace = {"_dim": cls._rank, "_rank": cls._rank, "_base": upd_base}
        if isinstance(upd_base, VectorBundle):
            if upd_base.incomplete:
                namespace.update(
                    {"__class_getitem__": VectorBundle._partial_spec_mfld_pass_to_base}
                )

        return PartialSpec(
            f"VectorBundle[{cls._rank}, {upd_base.__name__}]",
            (cls,),
            namespace,
            creating_derived=True,
            top_level_type=cls._top_level_type,
        )

    @classproperty
    def rank(cls):
        return cls._rank

    @classproperty
    def base(cls):
        return cls._base

    @classproperty
    def root(cls):
        # finds the underlying root of all the chained bundles
        if cls._base is not None and issubclass(cls._base, VectorBundle):
            return cls._base.root
        return cls._base

    @classproperty
    def bundles(cls):
        return (cls,)

    @classproperty
    def bundle_indices(cls):
        return {cls: 0}


class DualBundle(VectorBundle):
    _orig: type[VectorBundle]

    @classmethod
    def __class_getitem__(cls, args):
        orig: type[VectorBundle] = args

        if orig.rank == 0:
            raise TypeError("no dual exists for a vector bundle of rank 0")

        namespace = {
            "_dim": orig.dim,
            "_rank": orig.rank,
            "_base": orig.base,
            "_orig": orig,
        }
        if orig.incomplete:
            namespace.update(
                {"__class_getitem__": DualBundle._partial_spec_mfld_pass_to_base}
            )

        return PartialSpec(
            f"DualBundle[{orig.__name__}]",
            (DualBundle,),
            namespace,
            creating_derived=True,
        )

    def _partial_spec_mfld_pass_to_base(cls, args):
        underlying_base: type[Manifold] = args

        upd_orig = cls._orig[underlying_base]
        namespace = {
            "_dim": upd_orig.dim,
            "_rank": upd_orig.rank,
            "_base": underlying_base,
            "_orig": upd_orig,
        }

        return PartialSpec(
            f"DualBundle[{upd_orig}]",
            (DualBundle,),
            namespace,
            creating_derived=True,
            top_level_type=cls._top_level_type,
        )


class ScalarBundle(VectorBundle[0]):
    @classmethod
    def __class_getitem__(cls, args):
        base: type[Manifold] = args
        namespace = {"_dim": 0, "_rank": 0, "_base": base}

        if issubclass(base, VectorBundle):
            if base.incomplete:
                namespace.update(
                    {"__class_getitem__": ScalarBundle._partial_spec_mfld_pass_to_base}
                )

        return PartialSpec(
            f"ScalarBundle[{base.__name__}]",
            (cls,),
            namespace,
            creating_derived=True,
        )

    def _partial_spec_mfld_pass_to_base(cls, args):
        underlying_base: type[Manifold] = args

        upd_base = cls._base[underlying_base]
        namespace = {"_dim": 0, "_rank": 0, "_base": upd_base}
        if isinstance(upd_base, VectorBundle):
            if upd_base.incomplete:
                namespace.update(
                    {"__class_getitem__": ScalarBundle._partial_spec_mfld_pass_to_base}
                )

        return PartialSpec(
            f"ScalarBundle[{upd_base.__name__}]",
            (cls,),
            namespace,
            creating_derived=True,
            top_level_type=cls._top_level_type,
        )


class TangentBundle(VectorBundle):
    @classmethod
    def __class_getitem__(cls, args):
        base: type[Manifold] = args
        print(f"base: {base}")

        namespace = {"_dim": base.dim, "_rank": base.dim, "_base": base}

        # print("here")
        # print(base)

        if issubclass(base, VectorBundle):
            # print("is bundle")
            # print(f"is incomplete: {base.incomplete}")
            if base.incomplete:
                namespace.update(
                    {"__class_getitem__": TangentBundle._partial_spec_mfld_pass_to_base}
                )

        return PartialSpec(
            f"TangentBundle[{base.__name__}]",
            (TangentBundle,),
            namespace,
            creating_derived=True,
        )

    def _partial_spec_mfld_pass_to_base(cls, args):
        underlying_base: type[Manifold] = args
        upd_base = cls._base[underlying_base]

        # print(f"upd_base: {upd_base}")

        namespace = {"_dim": upd_base.dim, "_rank": upd_base.dim, "_base": upd_base}
        if issubclass(upd_base, VectorBundle):
            if upd_base.incomplete:
                namespace.update(
                    {"__class_getitem__": TangentBundle._partial_spec_mfld_pass_to_base}
                )

        return PartialSpec(
            f"TangentBundle[{upd_base.__name__}]",
            (TangentBundle,),
            namespace,
            creating_derived=True,
            top_level_type=cls._top_level_type,
        )


CotangentBundle = DualBundle[TangentBundle]


class TensorProductBundle(VectorBundle):
    _bundles: Tuple[VectorBundle, ...]

    def _gen_cls_name(bundles: Tuple[VectorBundle, ...]) -> str:
        bundles_comb = ""
        for i, bundle in enumerate(bundles):
            bundles_comb += bundle.__name__
            if i < len(bundles) - 1:
                bundles_comb += ", "

        return f"TensorProductBundle[{bundles_comb}]"

    def __class_getitem__(cls, args):
        if type(args) is not tuple:
            return args  # not a tensor if only provided a single bundle

        # must ensure that all the provided bundles are defined and share the same underlying manifold or that all are
        # undefined so that some base manifold can be provided through delayed specification
        bundles: Tuple[VectorBundle, ...] = args

        bundles_incomplete: Optional[bool] = None
        for bundle in bundles:
            if bundles_incomplete is None:
                bundles_incomplete = bundle.incomplete
            elif bundle.incomplete != bundles_incomplete:
                raise TypeError(
                    "all bundles combined through the tensor product must be all completely specialized or incomplete"
                )

        # if fully specialized then check to ensure that they share the same base manifold
        shared_base: Optional[type[Manifold]] = None
        if not bundles_incomplete:
            for bundle in bundles:
                if shared_base is None:
                    shared_base = bundle.base
                elif bundle.root != shared_base:
                    raise TypeError(
                        "all specialized bundles must share same root manifold"
                    )

        vs_rank = (
            prod(bundle.dim for bundle in bundles) if not bundles_incomplete else None
        )

        namespace = {
            "_dim": vs_rank,
            "_rank": vs_rank,
            "_base": shared_base,
            "_bundles": bundles,
        }
        if bundles_incomplete:
            namespace.update(
                {
                    "__class_getitem__": TensorProductBundle._partial_spec_mfld_pass_to_base
                }
            )

        return PartialSpec(
            cls._gen_cls_name(bundles),
            (TensorProductBundle,),
            namespace,
            creating_derived=True,
        )

    def _partial_spec_mfld_pass_to_base(cls, args):
        shared_base: type[Manifold] = args

        upd_bundles = tuple(bundle[shared_base] for bundle in cls._bundles)

        # print(f"upd_bundles: {upd_bundles}")

        vs_rank = prod(bundle.dim for bundle in upd_bundles)

        namespace = {
            "_dim": vs_rank,
            "_rank": vs_rank,
            "_base": shared_base,
            "_bundles": upd_bundles,
        }
        return PartialSpec(
            TensorProductBundle._gen_cls_name(upd_bundles),
            (TensorProductBundle,),
            namespace,
            creating_derived=True,
            top_level_type=cls._top_level_type,
        )

    @classproperty
    def bundles(cls):
        all_bundles = []
        for bundle in cls._bundles:
            all_bundles.extend(bundle.bundles)
        return tuple(all_bundles)

    @abstractmethod
    @classproperty
    def bundle_indices(cls):
        all_bundles = list(cls.bundles)
        unique_bundle_types = set(all_bundles)

        bundle_type_count = {
            unique_bundle: [
                i for i, bundle in enumerate(all_bundles) if bundle == unique_bundle
            ]
            for unique_bundle in unique_bundle_types
        }
        return bundle_type_count


class KBundle(TensorProductBundle):
    _bundle: type[VectorBundle]
    _copies: int

    @classmethod
    def __class_getitem__(cls, args):
        copies: int
        bundle: type[VectorBundle]
        copies, bundle = args

        print(f"copies: {copies}")
        print(f"bundle: {bundle}")

        vs_rank = bundle.dim * copies if not bundle.incomplete else None

        namespace = {
            "_dim": vs_rank,
            "_rank": vs_rank,
            "_base": bundle.base,
            "_bundle": bundle,
            "_copies": copies,
            "_bundles": tuple(bundle for _ in range(copies)),
        }
        if bundle.incomplete:
            namespace.update(
                {"__class_getitem__": KBundle._partial_spec_mfld_pass_to_base}
            )

        return PartialSpec(
            f"KBundle[{copies}, {bundle}]", (KBundle,), namespace, creating_derived=True
        )

    def _partial_spec_mfld_pass_to_base(cls, args):
        underlying_base: type[Manifold] = args
        upd_bundle = cls._bundle[underlying_base]
        return KBundle[cls._copies, upd_bundle]

    @classproperty
    def bundle(cls):
        return cls._bundle

    @classproperty
    def copies(cls):
        return cls._copies


class TensorBundle(TensorProductBundle):
    _contravariant: KBundle
    _covariant: KBundle

    @classmethod
    def __class_getitem__(cls, args):
        contravariant_rank: int
        covariant_rank: int
        base: Optional[type[Manifold]] = None

        print("here1")

        if len(args) == 2:
            contravariant_rank, covariant_rank = args
        else:
            contravariant_rank, covariant_rank, base = args

        contravariant = KBundle[contravariant_rank, TangentBundle]
        covariant = KBundle[covariant_rank, DualBundle[TangentBundle]]

        combined_bundles = []

        print(f"contravariant bundles: {contravariant.bundles}")
        print(f"covariant bundles: {covariant.bundles}")

        combined_bundles.extend(list(contravariant.bundles))
        combined_bundles.extend(list(covariant.bundles))

        if base is None or base.incomplete:
            namespace = {
                "_dim": None,
                "_rank": None,
                "_base": base,
                "_bundles": combined_bundles,
                "_contravariant": contravariant,
                "_covariant": covariant,
                "__class_getitem__": TensorBundle._partial_spec_mfld_pass_to_base,
            }
        else:
            contravariant = contravariant[base]
            covariant = covariant[base]

            vs_rank = base.dim * (contravariant.copies + covariant.copies)

            namespace = {
                "_dim": vs_rank,
                "_rank": vs_rank,
                "_base": base,
                "_bundles": combined_bundles,
                "_contravariant": contravariant,
                "_covariant": covariant,
            }

        return PartialSpec(
            f"TensorBundle[{contravariant_rank}, {covariant_rank}, {base}]",
            (TensorBundle,),
            namespace,
            creating_derived=True,
        )

    def _partial_spec_mfld_pass_to_base(cls, args):
        underlying_base: type[Manifold] = args

        print("here2")

        upd_contravariant: KBundle = cls._contravariant[underlying_base]
        upd_covariant: KBundle = cls._covariant[underlying_base]

        if underlying_base.incomplete:
            namespace = {
                "_dim": None,
                "_rank": None,
                "_base": underlying_base,
                "_bundles": [*upd_contravariant.bundles, *upd_covariant.bundles],
                "_contravariant": upd_contravariant,
                "_covariant": upd_covariant,
                "__class_getitem__": TensorBundle._partial_spec_mfld_pass_to_base,
            }
        else:
            vs_rank = underlying_base.dim * (
                upd_contravariant.copies + upd_covariant.copies
            )

            namespace = {
                "_dim": vs_rank,
                "_rank": vs_rank,
                "_base": underlying_base,
                "_bundles": [*upd_contravariant.bundles, *upd_covariant.bundles],
                "_contravariant": upd_contravariant,
                "_covariant": upd_covariant,
            }

        return PartialSpec(
            f"TensorBundle[{upd_contravariant.copies}, {upd_covariant.copies}, {underlying_base}]",
            (TensorBundle,),
            namespace,
            creating_derived=True,
            top_level_type=cls._top_level_type,
        )

    @classproperty
    def ctv_rank(cls):
        return cls._contravariant.copies

    @classproperty
    def cov_rank(cls):
        return cls._covariant.copies
