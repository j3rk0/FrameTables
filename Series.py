from datetime import timedelta
from typing import Tuple, Any, Type, Hashable, Callable, Sequence, overload, Literal, NoReturn, IO, Mapping, Iterable
from typing_extensions import SupportsIndex

import numpy as np
import pandas as pd
from pandas import DataFrame, Series, HDFStore, Index, MultiIndex, Flags
from pandas._libs import lib
from pandas._libs.tslibs import BaseOffset
from pandas._typing import NDFrameT, TimedeltaConvertibleTypes, Axis, Level, IntervalClosedType, IgnoreRaise, T, \
    RandomState, IndexLabel, FilePath, WriteBuffer, CompressionOptions, StorageOptions, ColspaceArgType, FormattersType, \
    FloatFormatType, DtypeArg, JSONSerializable, ArrayLike, Renamer, DtypeObj, Manager, Dtype, npt, Shape, \
    TimestampConvertibleTypes, Frequency, FillnaOptions, AggFuncType, SortKind, NaPosition, IndexKeyFunc, ValueKeyFunc, \
    Suffixes, NumpyValueArrayLike, NumpySorter, AnyArrayLike, QuantileInterpolation
from pandas.core.arrays import ExtensionArray
from pandas.core.base import _T
from pandas.core.generic import bool_t, NDFrame
from pandas.core.groupby import SeriesGroupBy
from pandas.core.indexers.objects import BaseIndexer
from pandas.core.indexing import _iAtIndexer, _AtIndexer, _LocIndexer, _iLocIndexer
from pandas.core.resample import Resampler
from pandas.core.window import ExponentialMovingWindow, Expanding, Window, Rolling


class Series(pd.Series):
    def __init__(self, data=None, index=None, dtype: Dtype | None = None, name=None, copy: bool = False,
                 fastpath: bool = False) -> None:
        super().__init__(data, index, dtype, name, copy, fastpath)

    def _init_dict(self, data, index: Index | None = None, dtype: DtypeObj | None = None):
        return super()._init_dict(data, index, dtype)

    @property
    def _constructor(self) -> Callable[..., Series]:
        return super()._constructor

    @property
    def _constructor_expanddim(self) -> Callable[..., DataFrame]:
        return super()._constructor_expanddim

    @property
    def _can_hold_na(self) -> bool:
        return super()._can_hold_na

    def _set_axis(self, axis: int, labels: AnyArrayLike | list) -> None:
        super()._set_axis(axis, labels)

    @property
    def dtype(self) -> DtypeObj:
        return super().dtype

    @property
    def dtypes(self) -> DtypeObj:
        return super().dtypes

    @property
    def name(self) -> Hashable:
        return super().name

    @property
    def values(self):
        return super().values

    @property
    def _values(self):
        return super()._values

    @property
    def array(self) -> ExtensionArray:
        return super().array

    def ravel(self, order: str = "C") -> np.ndarray:
        return super().ravel(order)

    def __len__(self) -> int:
        return super().__len__()

    def view(self, dtype: Dtype | None = None) -> Series:
        return super().view(dtype)

    def __array__(self, dtype: npt.DTypeLike | None = None) -> np.ndarray:
        return super().__array__(dtype)

    @property
    def axes(self) -> list[Index]:
        return super().axes

    def take(self, indices, axis: Axis = 0, is_copy: bool | None = None, **kwargs) -> Series:
        return super().take(indices, axis, is_copy, **kwargs)

    def _take_with_is_copy(self, indices, axis=0) -> Series:
        return super()._take_with_is_copy(indices, axis)

    def _ixs(self, i: int, axis: int = 0) -> Any:
        return super()._ixs(i, axis)

    def _slice(self, slobj: slice, axis: int = 0) -> Series:
        return super()._slice(slobj, axis)

    def __getitem__(self, key):
        return super().__getitem__(key)

    def _get_with(self, key):
        return super()._get_with(key)

    def _get_values_tuple(self, key: tuple):
        return super()._get_values_tuple(key)

    def _get_values(self, indexer: slice | npt.NDArray[np.bool_]) -> Series:
        return super()._get_values(indexer)

    def _get_value(self, label, takeable: bool = False):
        return super()._get_value(label, takeable)

    def __setitem__(self, key, value) -> None:
        super().__setitem__(key, value)

    def _set_with_engine(self, key, value) -> None:
        super()._set_with_engine(key, value)

    def _set_with(self, key, value):
        super()._set_with(key, value)

    def _set_labels(self, key, value) -> None:
        super()._set_labels(key, value)

    def _set_values(self, key, value) -> None:
        super()._set_values(key, value)

    def _set_value(self, label, value, takeable: bool = False):
        super()._set_value(label, value, takeable)

    @property
    def _is_cached(self) -> bool:
        return super()._is_cached

    def _get_cacher(self):
        return super()._get_cacher()

    def _reset_cacher(self) -> None:
        super()._reset_cacher()

    def _set_as_cached(self, item, cacher) -> None:
        super()._set_as_cached(item, cacher)

    def _clear_item_cache(self) -> None:
        super()._clear_item_cache()

    def _check_is_chained_assignment_possible(self) -> bool:
        return super()._check_is_chained_assignment_possible()

    def _maybe_update_cacher(self, clear: bool = False, verify_is_copy: bool = True, inplace: bool = False) -> None:
        super()._maybe_update_cacher(clear, verify_is_copy, inplace)

    @property
    def _is_mixed_type(self):
        return super()._is_mixed_type

    def repeat(self, repeats: int | Sequence[int], axis: None = None) -> Series:
        return super().repeat(repeats, axis)

    @overload
    def reset_index(
        self,
        level: IndexLabel = ...,
        *,
        drop: Literal[False] = ...,
        name: Level = ...,
        inplace: Literal[False] = ...,
        allow_duplicates: bool = ...,
    ) -> DataFrame:
        ...

    @overload
    def reset_index(
        self,
        level: IndexLabel = ...,
        *,
        drop: Literal[True],
        name: Level = ...,
        inplace: Literal[False] = ...,
        allow_duplicates: bool = ...,
    ) -> Series:
        ...

    @overload
    def reset_index(
        self,
        level: IndexLabel = ...,
        *,
        drop: bool = ...,
        name: Level = ...,
        inplace: Literal[True],
        allow_duplicates: bool = ...,
    ) -> None:
        ...

    def reset_index(self, level: IndexLabel = None, drop: bool = False, name: Level = lib.no_default,
                    inplace: bool = False, allow_duplicates: bool = False) -> DataFrame | Series | None:
        return super().reset_index(level, drop, name, inplace, allow_duplicates)

    def __repr__(self) -> str:
        return super().__repr__()

    @overload
    def to_string(
        self,
        buf: None = ...,
        na_rep: str = ...,
        float_format: str | None = ...,
        header: bool = ...,
        index: bool = ...,
        length=...,
        dtype=...,
        name=...,
        max_rows: int | None = ...,
        min_rows: int | None = ...,
    ) -> str:
        ...

    @overload
    def to_string(
        self,
        buf: FilePath | WriteBuffer[str],
        na_rep: str = ...,
        float_format: str | None = ...,
        header: bool = ...,
        index: bool = ...,
        length=...,
        dtype=...,
        name=...,
        max_rows: int | None = ...,
        min_rows: int | None = ...,
    ) -> None:
        ...

    def to_string(self, buf: FilePath | WriteBuffer[str] | None = None, na_rep: str = "NaN",
                  float_format: str | None = None, header: bool = True, index: bool = True, length=False, dtype=False,
                  name=False, max_rows: int | None = None, min_rows: int | None = None) -> str | None:
        return super().to_string(buf, na_rep, float_format, header, index, length, dtype, name, max_rows, min_rows)

    def to_markdown(self, buf: IO[str] | None = None, mode: str = "wt", index: bool = True,
                    storage_options: StorageOptions = None, **kwargs) -> str | None:
        return super().to_markdown(buf, mode, index, storage_options, **kwargs)

    def items(self) -> Iterable[tuple[Hashable, Any]]:
        return super().items()

    def iteritems(self) -> Iterable[tuple[Hashable, Any]]:
        return super().iteritems()

    def keys(self) -> Index:
        return super().keys()

    def to_dict(self, into: type[dict] = dict) -> dict:
        return super().to_dict(into)

    def to_frame(self, name: Hashable = lib.no_default) -> DataFrame:
        return super().to_frame(name)

    def _set_name(self, name, inplace=False) -> Series:
        return super()._set_name(name, inplace)

    def groupby(self, by=None, axis: Axis = 0, level: Level = None, as_index: bool = True, sort: bool = True,
                group_keys: bool | lib.NoDefault = no_default, squeeze: bool | lib.NoDefault = no_default,
                observed: bool = False, dropna: bool = True) -> SeriesGroupBy:
        return super().groupby(by, axis, level, as_index, sort, group_keys, squeeze, observed, dropna)

    def count(self, level: Level = None):
        return super().count(level)

    def mode(self, dropna: bool = True) -> Series:
        return super().mode(dropna)

    def unique(self) -> ArrayLike:
        return super().unique()

    @overload
    def drop_duplicates(
        self,
        keep: Literal["first", "last", False] = ...,
        *,
        inplace: Literal[False] = ...,
    ) -> Series:
        ...

    @overload
    def drop_duplicates(
        self, keep: Literal["first", "last", False] = ..., *, inplace: Literal[True]
    ) -> None:
        ...

    @overload
    def drop_duplicates(
        self, keep: Literal["first", "last", False] = ..., *, inplace: bool = ...
    ) -> Series | None:
        ...

    def drop_duplicates(self, keep: Literal["first", "last", False] = "first", inplace=False) -> Series | None:
        return super().drop_duplicates(keep, inplace)

    def duplicated(self, keep: Literal["first", "last", False] = "first") -> Series:
        return super().duplicated(keep)

    def idxmin(self, axis: Axis = 0, skipna: bool = True, *args, **kwargs) -> Hashable:
        return super().idxmin(axis, skipna, *args, **kwargs)

    def idxmax(self, axis: Axis = 0, skipna: bool = True, *args, **kwargs) -> Hashable:
        return super().idxmax(axis, skipna, *args, **kwargs)

    def round(self, decimals: int = 0, *args, **kwargs) -> Series:
        return super().round(decimals, *args, **kwargs)

    @overload
    def quantile(
        self, q: float = ..., interpolation: QuantileInterpolation = ...
    ) -> float:
        ...

    @overload
    def quantile(
        self,
        q: Sequence[float] | AnyArrayLike,
        interpolation: QuantileInterpolation = ...,
    ) -> Series:
        ...

    @overload
    def quantile(
        self,
        q: float | Sequence[float] | AnyArrayLike = ...,
        interpolation: QuantileInterpolation = ...,
    ) -> float | Series:
        ...

    def quantile(self, q: float | Sequence[float] | AnyArrayLike = 0.5,
                 interpolation: QuantileInterpolation = "linear") -> float | Series:
        return super().quantile(q, interpolation)

    def corr(self, other: Series, method: Literal["pearson", "kendall", "spearman"]
                                          | Callable[[np.ndarray, np.ndarray], float] = "pearson",
             min_periods: int | None = None) -> float:
        return super().corr(other, method, min_periods)

    def cov(self, other: Series, min_periods: int | None = None, ddof: int | None = 1) -> float:
        return super().cov(other, min_periods, ddof)

    def diff(self, periods: int = 1) -> Series:
        return super().diff(periods)

    def autocorr(self, lag: int = 1) -> float:
        return super().autocorr(lag)

    def dot(self, other: AnyArrayLike) -> Series | np.ndarray:
        return super().dot(other)

    def __matmul__(self, other):
        return super().__matmul__(other)

    def __rmatmul__(self, other):
        return super().__rmatmul__(other)

    def searchsorted(self, value: NumpyValueArrayLike | ExtensionArray, side: Literal["left", "right"] = "left",
                     sorter: NumpySorter = None) -> npt.NDArray[np.intp] | np.intp:
        return super().searchsorted(value, side, sorter)

    def append(self, to_append, ignore_index: bool = False, verify_integrity: bool = False) -> Series:
        return super().append(to_append, ignore_index, verify_integrity)

    def _append(self, to_append, ignore_index: bool = False, verify_integrity: bool = False):
        return super()._append(to_append, ignore_index, verify_integrity)

    def _binop(self, other: Series, func, level=None, fill_value=None):
        return super()._binop(other, func, level, fill_value)

    def _construct_result(self, result: ArrayLike | tuple[ArrayLike, ArrayLike], name: Hashable) -> Series | tuple[
        Series, Series]:
        return super()._construct_result(result, name)

    def compare(self, other: Series, align_axis: Axis = 1, keep_shape: bool = False, keep_equal: bool = False,
                result_names: Suffixes = ("self", "other")) -> DataFrame | Series:
        return super().compare(other, align_axis, keep_shape, keep_equal, result_names)

    def combine(self, other: Series | Hashable, func: Callable[[Hashable, Hashable], Hashable],
                fill_value: Hashable = None) -> Series:
        return super().combine(other, func, fill_value)

    def combine_first(self, other) -> Series:
        return super().combine_first(other)

    def update(self, other: Series | Sequence | Mapping) -> None:
        super().update(other)

    @overload  # type: ignore[override]
    def sort_values(
        self,
        *,
        axis: Axis = ...,
        ascending: bool | int | Sequence[bool] | Sequence[int] = ...,
        inplace: Literal[False] = ...,
        kind: str = ...,
        na_position: str = ...,
        ignore_index: bool = ...,
        key: ValueKeyFunc = ...,
    ) -> Series:
        ...

    @overload
    def sort_values(
        self,
        *,
        axis: Axis = ...,
        ascending: bool | int | Sequence[bool] | Sequence[int] = ...,
        inplace: Literal[True],
        kind: str = ...,
        na_position: str = ...,
        ignore_index: bool = ...,
        key: ValueKeyFunc = ...,
    ) -> None:
        ...

    def sort_values(self, axis: Axis = 0, ascending: bool | int | Sequence[bool] | Sequence[int] = True,
                    inplace: bool = False, kind: str = "quicksort", na_position: str = "last",
                    ignore_index: bool = False, key: ValueKeyFunc = None) -> Series | None:
        return super().sort_values(axis, ascending, inplace, kind, na_position, ignore_index, key)

    @overload
    def sort_index(
        self,
        *,
        axis: Axis = ...,
        level: IndexLabel = ...,
        ascending: bool | Sequence[bool] = ...,
        inplace: Literal[True],
        kind: SortKind = ...,
        na_position: NaPosition = ...,
        sort_remaining: bool = ...,
        ignore_index: bool = ...,
        key: IndexKeyFunc = ...,
    ) -> None:
        ...

    @overload
    def sort_index(
        self,
        *,
        axis: Axis = ...,
        level: IndexLabel = ...,
        ascending: bool | Sequence[bool] = ...,
        inplace: Literal[False] = ...,
        kind: SortKind = ...,
        na_position: NaPosition = ...,
        sort_remaining: bool = ...,
        ignore_index: bool = ...,
        key: IndexKeyFunc = ...,
    ) -> Series:
        ...

    @overload
    def sort_index(
        self,
        *,
        axis: Axis = ...,
        level: IndexLabel = ...,
        ascending: bool | Sequence[bool] = ...,
        inplace: bool = ...,
        kind: SortKind = ...,
        na_position: NaPosition = ...,
        sort_remaining: bool = ...,
        ignore_index: bool = ...,
        key: IndexKeyFunc = ...,
    ) -> Series | None:
        ...

    def sort_index(self, axis: Axis = 0, level: IndexLabel = None, ascending: bool | Sequence[bool] = True,
                   inplace: bool = False, kind: SortKind = "quicksort", na_position: NaPosition = "last",
                   sort_remaining: bool = True, ignore_index: bool = False, key: IndexKeyFunc = None) -> Series | None:
        return super().sort_index(axis, level, ascending, inplace, kind, na_position, sort_remaining, ignore_index, key)

    def argsort(self, axis: Axis = 0, kind: SortKind = "quicksort", order: None = None) -> Series:
        return super().argsort(axis, kind, order)

    def nlargest(self, n: int = 5, keep: Literal["first", "last", "all"] = "first") -> Series:
        return super().nlargest(n, keep)

    def nsmallest(self, n: int = 5, keep: str = "first") -> Series:
        return super().nsmallest(n, keep)

    def swaplevel(self, i: Level = -2, j: Level = -1, copy: bool = True) -> Series:
        return super().swaplevel(i, j, copy)

    def reorder_levels(self, order: Sequence[Level]) -> Series:
        return super().reorder_levels(order)

    def explode(self, ignore_index: bool = False) -> Series:
        return super().explode(ignore_index)

    def unstack(self, level: IndexLabel = -1, fill_value: Hashable = None) -> DataFrame:
        return super().unstack(level, fill_value)

    def map(self, arg: Callable | Mapping | Series, na_action: Literal["ignore"] | None = None) -> Series:
        return super().map(arg, na_action)

    def _gotitem(self, key, ndim, subset=None) -> Series:
        return super()._gotitem(key, ndim, subset)

    def aggregate(self, func=None, axis: Axis = 0, *args, **kwargs):
        return super().aggregate(func, axis, *args, **kwargs)

    @overload  # type: ignore[override]
    def any(
        self,
        *,
        axis: Axis = ...,
        bool_only: bool | None = ...,
        skipna: bool = ...,
        level: None = ...,
        **kwargs,
    ) -> bool:
        ...

    @overload
    def any(
        self,
        *,
        axis: Axis = ...,
        bool_only: bool | None = ...,
        skipna: bool = ...,
        level: Level,
        **kwargs,
    ) -> Series | bool:
        ...

    def any(self, axis: Axis = 0, bool_only: bool | None = None, skipna: bool = True, level: Level | None = None,
            **kwargs) -> Series | bool:
        return super().any(axis, bool_only, skipna, level, **kwargs)

    def transform(self, func: AggFuncType, axis: Axis = 0, *args, **kwargs) -> DataFrame | Series:
        return super().transform(func, axis, *args, **kwargs)

    def apply(self, func: AggFuncType, convert_dtype: bool = True, args: tuple[Any, ...] = (),
              **kwargs) -> DataFrame | Series:
        return super().apply(func, convert_dtype, args, **kwargs)

    def _reduce(self, op, name: str, *, axis=0, skipna=True, numeric_only=None, filter_type=None, **kwds):
        return super()._reduce(op, name, axis=axis, skipna=skipna, numeric_only=numeric_only, filter_type=filter_type,
                               **kwds)

    def _reindex_indexer(self, new_index: Index | None, indexer: npt.NDArray[np.intp] | None, copy: bool) -> Series:
        return super()._reindex_indexer(new_index, indexer, copy)

    def _needs_reindex_multi(self, axes, method, level) -> bool:
        return super()._needs_reindex_multi(axes, method, level)

    def align(self, other: Series, join: Literal["outer", "inner", "left", "right"] = "outer", axis: Axis | None = None,
              level: Level = None, copy: bool = True, fill_value: Hashable = None, method: FillnaOptions | None = None,
              limit: int | None = None, fill_axis: Axis = 0, broadcast_axis: Axis | None = None) -> Series:
        return super().align(other, join, axis, level, copy, fill_value, method, limit, fill_axis, broadcast_axis)

    @overload
    def rename(
        self,
        index: Renamer | Hashable | None = ...,
        *,
        axis: Axis | None = ...,
        copy: bool = ...,
        inplace: Literal[True],
        level: Level | None = ...,
        errors: IgnoreRaise = ...,
    ) -> None:
        ...

    @overload
    def rename(
        self,
        index: Renamer | Hashable | None = ...,
        *,
        axis: Axis | None = ...,
        copy: bool = ...,
        inplace: Literal[False] = ...,
        level: Level | None = ...,
        errors: IgnoreRaise = ...,
    ) -> Series:
        ...

    @overload
    def rename(
        self,
        index: Renamer | Hashable | None = ...,
        *,
        axis: Axis | None = ...,
        copy: bool = ...,
        inplace: bool = ...,
        level: Level | None = ...,
        errors: IgnoreRaise = ...,
    ) -> Series | None:
        ...

    def rename(self, index: Renamer | Hashable | None = None, *, axis: Axis | None = None, copy: bool = True,
               inplace: bool = False, level: Level | None = None, errors: IgnoreRaise = "ignore") -> Series | None:
        return super().rename(index, axis=axis, copy=copy, inplace=inplace, level=level, errors=errors)

    @overload
    def set_axis(
        self,
        labels,
        *,
        axis: Axis = ...,
        inplace: Literal[False] | lib.NoDefault = ...,
        copy: bool | lib.NoDefault = ...,
    ) -> Series:
        ...

    @overload
    def set_axis(
        self,
        labels,
        *,
        axis: Axis = ...,
        inplace: Literal[True],
        copy: bool | lib.NoDefault = ...,
    ) -> None:
        ...

    @overload
    def set_axis(
        self,
        labels,
        *,
        axis: Axis = ...,
        inplace: bool | lib.NoDefault = ...,
        copy: bool | lib.NoDefault = ...,
    ) -> Series | None:
        ...

    def set_axis(self, labels, axis: Axis = 0, inplace: bool | lib.NoDefault = lib.no_default,
                 copy: bool | lib.NoDefault = lib.no_default) -> Series | None:
        return super().set_axis(labels, axis, inplace, copy)

    def reindex(self, *args, **kwargs) -> Series:
        return super().reindex(*args, **kwargs)

    @overload
    def drop(
        self,
        labels: IndexLabel = ...,
        *,
        axis: Axis = ...,
        index: IndexLabel = ...,
        columns: IndexLabel = ...,
        level: Level | None = ...,
        inplace: Literal[True],
        errors: IgnoreRaise = ...,
    ) -> None:
        ...

    @overload
    def drop(
        self,
        labels: IndexLabel = ...,
        *,
        axis: Axis = ...,
        index: IndexLabel = ...,
        columns: IndexLabel = ...,
        level: Level | None = ...,
        inplace: Literal[False] = ...,
        errors: IgnoreRaise = ...,
    ) -> Series:
        ...

    @overload
    def drop(
        self,
        labels: IndexLabel = ...,
        *,
        axis: Axis = ...,
        index: IndexLabel = ...,
        columns: IndexLabel = ...,
        level: Level | None = ...,
        inplace: bool = ...,
        errors: IgnoreRaise = ...,
    ) -> Series | None:
        ...

    def drop(self, labels: IndexLabel = None, axis: Axis = 0, index: IndexLabel = None, columns: IndexLabel = None,
             level: Level | None = None, inplace: bool = False, errors: IgnoreRaise = "raise") -> Series | None:
        return super().drop(labels, axis, index, columns, level, inplace, errors)

    @overload
    def fillna(
        self,
        value: Hashable | Mapping | Series | DataFrame = ...,
        *,
        method: FillnaOptions | None = ...,
        axis: Axis | None = ...,
        inplace: Literal[False] = ...,
        limit: int | None = ...,
        downcast: dict | None = ...,
    ) -> Series:
        ...

    @overload
    def fillna(
        self,
        value: Hashable | Mapping | Series | DataFrame = ...,
        *,
        method: FillnaOptions | None = ...,
        axis: Axis | None = ...,
        inplace: Literal[True],
        limit: int | None = ...,
        downcast: dict | None = ...,
    ) -> None:
        ...

    @overload
    def fillna(
        self,
        value: Hashable | Mapping | Series | DataFrame = ...,
        *,
        method: FillnaOptions | None = ...,
        axis: Axis | None = ...,
        inplace: bool = ...,
        limit: int | None = ...,
        downcast: dict | None = ...,
    ) -> Series | None:
        ...

    def fillna(self, value: Hashable | Mapping | Series | DataFrame = None, method: FillnaOptions | None = None,
               axis: Axis | None = None, inplace: bool = False, limit: int | None = None,
               downcast: dict | None = None) -> Series | None:
        return super().fillna(value, method, axis, inplace, limit, downcast)

    def pop(self, item: Hashable) -> Any:
        return super().pop(item)

    @overload  # type: ignore[override]
    def replace(
        self,
        to_replace=...,
        value=...,
        *,
        inplace: Literal[False] = ...,
        limit: int | None = ...,
        regex: bool = ...,
        method: Literal["pad", "ffill", "bfill"] | lib.NoDefault = ...,
    ) -> Series:
        ...

    @overload
    def replace(
        self,
        to_replace=...,
        value=...,
        *,
        inplace: Literal[True],
        limit: int | None = ...,
        regex: bool = ...,
        method: Literal["pad", "ffill", "bfill"] | lib.NoDefault = ...,
    ) -> None:
        ...

    def replace(self, to_replace=None, value=lib.no_default, inplace: bool = False, limit: int | None = None,
                regex: bool = False,
                method: Literal["pad", "ffill", "bfill"] | lib.NoDefault = lib.no_default) -> Series | None:
        return super().replace(to_replace, value, inplace, limit, regex, method)

    def info(self, verbose: bool | None = None, buf: IO[str] | None = None, max_cols: int | None = None,
             memory_usage: bool | str | None = None, show_counts: bool = True) -> None:
        super().info(verbose, buf, max_cols, memory_usage, show_counts)

    def _replace_single(self, to_replace, method: str, inplace: bool, limit):
        return super()._replace_single(to_replace, method, inplace, limit)

    def shift(self, periods: int = 1, freq=None, axis: Axis = 0, fill_value: Hashable = None) -> Series:
        return super().shift(periods, freq, axis, fill_value)

    def memory_usage(self, index: bool = True, deep: bool = False) -> int:
        return super().memory_usage(index, deep)

    def isin(self, values) -> Series:
        return super().isin(values)

    def between(self, left, right, inclusive: Literal["both", "neither", "left", "right"] = "both") -> Series:
        return super().between(left, right, inclusive)

    def _convert_dtypes(self, infer_objects: bool = True, convert_string: bool = True, convert_integer: bool = True,
                        convert_boolean: bool = True, convert_floating: bool = True) -> Series:
        return super()._convert_dtypes(infer_objects, convert_string, convert_integer, convert_boolean,
                                       convert_floating)

    def isna(self) -> Series:
        return super().isna()

    def isnull(self) -> Series:
        return super().isnull()

    def notna(self) -> Series:
        return super().notna()

    def notnull(self) -> Series:
        return super().notnull()

    @overload
    def dropna(
        self, *, axis: Axis = ..., inplace: Literal[False] = ..., how: str | None = ...
    ) -> Series:
        ...

    @overload
    def dropna(
        self, *, axis: Axis = ..., inplace: Literal[True], how: str | None = ...
    ) -> None:
        ...

    def dropna(self, axis: Axis = 0, inplace: bool = False, how: str | None = None) -> Series | None:
        return super().dropna(axis, inplace, how)

    def asfreq(self, freq: Frequency, method: FillnaOptions | None = None, how: str | None = None,
               normalize: bool = False, fill_value: Hashable = None) -> Series:
        return super().asfreq(freq, method, how, normalize, fill_value)

    def resample(self, rule, axis: Axis = 0, closed: str | None = None, label: str | None = None,
                 convention: str = "start", kind: str | None = None, loffset=None, base: int | None = None,
                 on: Level = None, level: Level = None, origin: str | TimestampConvertibleTypes = "start_day",
                 offset: TimedeltaConvertibleTypes | None = None,
                 group_keys: bool | lib.NoDefault = no_default) -> Resampler:
        return super().resample(rule, axis, closed, label, convention, kind, loffset, base, on, level, origin, offset,
                                group_keys)

    def to_timestamp(self, freq=None, how: Literal["s", "e", "start", "end"] = "start", copy: bool = True) -> Series:
        return super().to_timestamp(freq, how, copy)

    def to_period(self, freq: str | None = None, copy: bool = True) -> Series:
        return super().to_period(freq, copy)

    @overload
    def ffill(
        self,
        *,
        axis: None | Axis = ...,
        inplace: Literal[False] = ...,
        limit: None | int = ...,
        downcast: dict | None = ...,
    ) -> Series:
        ...

    @overload
    def ffill(
        self,
        *,
        axis: None | Axis = ...,
        inplace: Literal[True],
        limit: None | int = ...,
        downcast: dict | None = ...,
    ) -> None:
        ...

    @overload
    def ffill(
        self,
        *,
        axis: None | Axis = ...,
        inplace: bool = ...,
        limit: None | int = ...,
        downcast: dict | None = ...,
    ) -> Series | None:
        ...

    def ffill(self, axis: None | Axis = None, inplace: bool = False, limit: None | int = None,
              downcast: dict | None = None) -> Series | None:
        return super().ffill(axis, inplace, limit, downcast)

    @overload
    def bfill(
        self,
        *,
        axis: None | Axis = ...,
        inplace: Literal[False] = ...,
        limit: None | int = ...,
        downcast: dict | None = ...,
    ) -> Series:
        ...

    @overload
    def bfill(
        self,
        *,
        axis: None | Axis = ...,
        inplace: Literal[True],
        limit: None | int = ...,
        downcast: dict | None = ...,
    ) -> None:
        ...

    @overload
    def bfill(
        self,
        *,
        axis: None | Axis = ...,
        inplace: bool = ...,
        limit: None | int = ...,
        downcast: dict | None = ...,
    ) -> Series | None:
        ...

    def bfill(self, axis: None | Axis = None, inplace: bool = False, limit: None | int = None,
              downcast: dict | None = None) -> Series | None:
        return super().bfill(axis, inplace, limit, downcast)

    def clip(self: Series, lower=None, upper=None, axis: Axis | None = None, inplace: bool = False, *args,
             **kwargs) -> Series | None:
        return super().clip(lower, upper, axis, inplace, *args, **kwargs)

    def interpolate(self: Series, method: str = "linear", axis: Axis = 0, limit: int | None = None,
                    inplace: bool = False, limit_direction: str | None = None, limit_area: str | None = None,
                    downcast: str | None = None, **kwargs) -> Series | None:
        return super().interpolate(method, axis, limit, inplace, limit_direction, limit_area, downcast, **kwargs)

    @overload
    def where(
        self,
        cond,
        other=...,
        *,
        inplace: Literal[False] = ...,
        axis: Axis | None = ...,
        level: Level = ...,
        errors: IgnoreRaise | lib.NoDefault = ...,
        try_cast: bool | lib.NoDefault = ...,
    ) -> Series:
        ...

    @overload
    def where(
        self,
        cond,
        other=...,
        *,
        inplace: Literal[True],
        axis: Axis | None = ...,
        level: Level = ...,
        errors: IgnoreRaise | lib.NoDefault = ...,
        try_cast: bool | lib.NoDefault = ...,
    ) -> None:
        ...

    @overload
    def where(
        self,
        cond,
        other=...,
        *,
        inplace: bool = ...,
        axis: Axis | None = ...,
        level: Level = ...,
        errors: IgnoreRaise | lib.NoDefault = ...,
        try_cast: bool | lib.NoDefault = ...,
    ) -> Series | None:
        ...

    def where(self, cond, other=lib.no_default, inplace: bool = False, axis: Axis | None = None, level: Level = None,
              errors: IgnoreRaise | lib.NoDefault = lib.no_default,
              try_cast: bool | lib.NoDefault = lib.no_default) -> Series | None:
        return super().where(cond, other, inplace, axis, level, errors, try_cast)

    @overload
    def mask(
        self,
        cond,
        other=...,
        *,
        inplace: Literal[False] = ...,
        axis: Axis | None = ...,
        level: Level = ...,
        errors: IgnoreRaise | lib.NoDefault = ...,
        try_cast: bool | lib.NoDefault = ...,
    ) -> Series:
        ...

    @overload
    def mask(
        self,
        cond,
        other=...,
        *,
        inplace: Literal[True],
        axis: Axis | None = ...,
        level: Level = ...,
        errors: IgnoreRaise | lib.NoDefault = ...,
        try_cast: bool | lib.NoDefault = ...,
    ) -> None:
        ...

    @overload
    def mask(
        self,
        cond,
        other=...,
        *,
        inplace: bool = ...,
        axis: Axis | None = ...,
        level: Level = ...,
        errors: IgnoreRaise | lib.NoDefault = ...,
        try_cast: bool | lib.NoDefault = ...,
    ) -> Series | None:
        ...

    def mask(self, cond, other=np.nan, inplace: bool = False, axis: Axis | None = None, level: Level = None,
             errors: IgnoreRaise | lib.NoDefault = lib.no_default,
             try_cast: bool | lib.NoDefault = lib.no_default) -> Series | None:
        return super().mask(cond, other, inplace, axis, level, errors, try_cast)

    def _cmp_method(self, other, op):
        return super()._cmp_method(other, op)

    def _logical_method(self, other, op):
        return super()._logical_method(other, op)

    def _arith_method(self, other, op):
        return super()._arith_method(other, op)

    def transpose(self: _T, *args, **kwargs) -> _T:
        return super().transpose(*args, **kwargs)

    @property
    def shape(self) -> Shape:
        return super().shape

    @property
    def ndim(self) -> Literal[1]:
        return super().ndim

    def item(self):
        return super().item()

    @property
    def nbytes(self) -> int:
        return super().nbytes

    @property
    def size(self) -> int:
        return super().size

    def to_numpy(self, dtype: npt.DTypeLike | None = None, copy: bool = False, na_value: object = lib.no_default,
                 **kwargs) -> np.ndarray:
        return super().to_numpy(dtype, copy, na_value, **kwargs)

    @property
    def empty(self) -> bool:
        return super().empty

    def max(self, axis=None, skipna: bool = True, *args, **kwargs):
        return super().max(axis, skipna, *args, **kwargs)

    def argmax(self, axis=None, skipna: bool = True, *args, **kwargs) -> int:
        return super().argmax(axis, skipna, *args, **kwargs)

    def min(self, axis=None, skipna: bool = True, *args, **kwargs):
        return super().min(axis, skipna, *args, **kwargs)

    def argmin(self, axis=None, skipna=True, *args, **kwargs) -> int:
        return super().argmin(axis, skipna, *args, **kwargs)

    def tolist(self):
        return super().tolist()

    def __iter__(self):
        return super().__iter__()

    def hasnans(self) -> bool:
        return super().hasnans()

    def _map_values(self, mapper, na_action=None):
        return super()._map_values(mapper, na_action)

    def value_counts(self, normalize: bool = False, sort: bool = True, ascending: bool = False, bins=None,
                     dropna: bool = True) -> Series:
        return super().value_counts(normalize, sort, ascending, bins, dropna)

    def nunique(self, dropna: bool = True) -> int:
        return super().nunique(dropna)

    @property
    def is_unique(self) -> bool:
        return super().is_unique

    @property
    def is_monotonic(self) -> bool:
        return super().is_monotonic

    @property
    def is_monotonic_increasing(self) -> bool:
        return super().is_monotonic_increasing

    @property
    def is_monotonic_decreasing(self) -> bool:
        return super().is_monotonic_decreasing

    def _memory_usage(self, deep: bool = False) -> int:
        return super()._memory_usage(deep)

    def factorize(self, sort: bool = False, na_sentinel: int | lib.NoDefault = lib.no_default,
                  use_na_sentinel: bool | lib.NoDefault = lib.no_default):
        return super().factorize(sort, na_sentinel, use_na_sentinel)

    def _duplicated(self, keep: Literal["first", "last", False] = "first") -> npt.NDArray[np.bool_]:
        return super()._duplicated(keep)

    def __eq__(self, other):
        return super().__eq__(other)

    def __ne__(self, other):
        return super().__ne__(other)

    def __lt__(self, other):
        return super().__lt__(other)

    def __le__(self, other):
        return super().__le__(other)

    def __gt__(self, other):
        return super().__gt__(other)

    def __ge__(self, other):
        return super().__ge__(other)

    def __and__(self, other):
        return super().__and__(other)

    def __rand__(self, other):
        return super().__rand__(other)

    def __or__(self, other):
        return super().__or__(other)

    def __ror__(self, other):
        return super().__ror__(other)

    def __xor__(self, other):
        return super().__xor__(other)

    def __rxor__(self, other):
        return super().__rxor__(other)

    def __add__(self, other):
        return super().__add__(other)

    def __radd__(self, other):
        return super().__radd__(other)

    def __sub__(self, other):
        return super().__sub__(other)

    def __rsub__(self, other):
        return super().__rsub__(other)

    def __mul__(self, other):
        return super().__mul__(other)

    def __rmul__(self, other):
        return super().__rmul__(other)

    def __truediv__(self, other):
        return super().__truediv__(other)

    def __rtruediv__(self, other):
        return super().__rtruediv__(other)

    def __floordiv__(self, other):
        return super().__floordiv__(other)

    def __rfloordiv__(self, other):
        return super().__rfloordiv__(other)

    def __mod__(self, other):
        return super().__mod__(other)

    def __rmod__(self, other):
        return super().__rmod__(other)

    def __divmod__(self, other):
        return super().__divmod__(other)

    def __rdivmod__(self, other):
        return super().__rdivmod__(other)

    def __pow__(self, other):
        return super().__pow__(other)

    def __rpow__(self, other):
        return super().__rpow__(other)

    @classmethod
    def _init_mgr(cls, mgr: Manager, axes, dtype: Dtype | None = None, copy: bool_t = False) -> Manager:
        return super()._init_mgr(mgr, axes, dtype, copy)

    def _as_manager(self: NDFrameT, typ: str, copy: bool_t = True) -> NDFrameT:
        return super()._as_manager(typ, copy)

    @property
    def attrs(self) -> dict[Hashable, Any]:
        return super().attrs

    @property
    def flags(self) -> Flags:
        return super().flags

    def set_flags(self: NDFrameT, *, copy: bool_t = False, allows_duplicate_labels: bool_t | None = None) -> NDFrameT:
        return super().set_flags(copy=copy, allows_duplicate_labels=allows_duplicate_labels)

    @classmethod
    def _validate_dtype(cls, dtype) -> DtypeObj | None:
        return super()._validate_dtype(dtype)

    @property
    def _data(self):
        return super()._data

    @property
    def _AXIS_NUMBERS(self) -> dict[str, int]:
        return super()._AXIS_NUMBERS

    @property
    def _AXIS_NAMES(self) -> dict[int, str]:
        return super()._AXIS_NAMES

    def _construct_axes_dict(self, axes=None, **kwargs):
        return super()._construct_axes_dict(axes, **kwargs)

    @classmethod
    def _construct_axes_from_arguments(cls, args, kwargs, require_all: bool_t = False, sentinel=None):
        return super()._construct_axes_from_arguments(args, kwargs, require_all, sentinel)

    @classmethod
    def _get_axis_number(cls, axis: Axis) -> int:
        return super()._get_axis_number(axis)

    @classmethod
    def _get_axis_name(cls, axis: Axis) -> str:
        return super()._get_axis_name(axis)

    def _get_axis(self, axis: Axis) -> Index:
        return super()._get_axis(axis)

    @classmethod
    def _get_block_manager_axis(cls, axis: Axis) -> int:
        return super()._get_block_manager_axis(axis)

    def _get_axis_resolvers(self, axis: str) -> dict[str, Series | MultiIndex]:
        return super()._get_axis_resolvers(axis)

    def _get_index_resolvers(self) -> dict[Hashable, Series | MultiIndex]:
        return super()._get_index_resolvers()

    def _get_cleaned_column_resolvers(self) -> dict[Hashable, Series]:
        return super()._get_cleaned_column_resolvers()

    @property
    def _info_axis(self) -> Index:
        return super()._info_axis

    @property
    def _stat_axis(self) -> Index:
        return super()._stat_axis

    def _set_axis_nocheck(self, labels, axis: Axis, inplace: bool_t, copy: bool_t):
        return super()._set_axis_nocheck(labels, axis, inplace, copy)

    def swapaxes(self: NDFrameT, axis1: Axis, axis2: Axis, copy: bool_t = True) -> NDFrameT:
        return super().swapaxes(axis1, axis2, copy)

    def droplevel(self: NDFrameT, level: IndexLabel, axis: Axis = 0) -> NDFrameT:
        return super().droplevel(level, axis)

    def squeeze(self, axis=None):
        return super().squeeze(axis)

    def _rename(self: NDFrameT, mapper: Renamer | None = None, *, index: Renamer | None = None,
                columns: Renamer | None = None, axis: Axis | None = None, copy: bool_t | None = None,
                inplace: bool_t = False, level: Level | None = None, errors: str = "ignore") -> NDFrameT | None:
        return super()._rename(mapper, index=index, columns=columns, axis=axis, copy=copy, inplace=inplace, level=level,
                               errors=errors)

    @overload
    def rename_axis(
        self: NDFrameT,
        mapper: IndexLabel | lib.NoDefault = ...,
        *,
        inplace: Literal[False] = ...,
        **kwargs,
    ) -> NDFrameT:
        ...

    @overload
    def rename_axis(
        self,
        mapper: IndexLabel | lib.NoDefault = ...,
        *,
        inplace: Literal[True],
        **kwargs,
    ) -> None:
        ...

    @overload
    def rename_axis(
        self: NDFrameT,
        mapper: IndexLabel | lib.NoDefault = ...,
        *,
        inplace: bool_t = ...,
        **kwargs,
    ) -> NDFrameT | None:
        ...

    def rename_axis(self: NDFrameT, mapper: IndexLabel | lib.NoDefault = lib.no_default, inplace: bool_t = False,
                    **kwargs) -> NDFrameT | None:
        return super().rename_axis(mapper, inplace, **kwargs)

    def _set_axis_name(self, name, axis=0, inplace=False):
        return super()._set_axis_name(name, axis, inplace)

    def _indexed_same(self, other) -> bool_t:
        return super()._indexed_same(other)

    def equals(self, other: object) -> bool_t:
        return super().equals(other)

    def __neg__(self: NDFrameT) -> NDFrameT:
        return super().__neg__()

    def __pos__(self: NDFrameT) -> NDFrameT:
        return super().__pos__()

    def __invert__(self: NDFrameT) -> NDFrameT:
        return super().__invert__()

    def __nonzero__(self) -> NoReturn:
        return super().__nonzero__()

    def bool(self) -> bool_t:
        return super().bool()

    def abs(self: NDFrameT) -> NDFrameT:
        return super().abs()

    def __abs__(self: NDFrameT) -> NDFrameT:
        return super().__abs__()

    def __round__(self: NDFrameT, decimals: int = 0) -> NDFrameT:
        return super().__round__(decimals)

    def _is_level_reference(self, key: Level, axis=0) -> bool_t:
        return super()._is_level_reference(key, axis)

    def _is_label_reference(self, key: Level, axis=0) -> bool_t:
        return super()._is_label_reference(key, axis)

    def _is_label_or_level_reference(self, key: Level, axis: int = 0) -> bool_t:
        return super()._is_label_or_level_reference(key, axis)

    def _check_label_or_level_ambiguity(self, key: Level, axis: int = 0) -> None:
        super()._check_label_or_level_ambiguity(key, axis)

    def _get_label_or_level_values(self, key: Level, axis: int = 0) -> ArrayLike:
        return super()._get_label_or_level_values(key, axis)

    def _drop_labels_or_levels(self, keys, axis: int = 0):
        return super()._drop_labels_or_levels(keys, axis)

    def __contains__(self, key) -> bool_t:
        return super().__contains__(key)

    def __array_wrap__(self, result: np.ndarray, context: tuple[Callable, tuple[Any, ...], int] | None = None):
        return super().__array_wrap__(result, context)

    def __array_ufunc__(self, ufunc: np.ufunc, method: str, *inputs: Any, **kwargs: Any):
        return super().__array_ufunc__(ufunc, method, *inputs, **kwargs)

    def __getstate__(self) -> dict[str, Any]:
        return super().__getstate__()

    def __setstate__(self, state) -> None:
        super().__setstate__(state)

    def _repr_latex_(self):
        return super()._repr_latex_()

    def _repr_data_resource_(self):
        return super()._repr_data_resource_()

    def to_excel(self, excel_writer, sheet_name: str = "Sheet1", na_rep: str = "", float_format: str | None = None,
                 columns: Sequence[Hashable] | None = None, header: Sequence[Hashable] | bool_t = True,
                 index: bool_t = True, index_label: IndexLabel = None, startrow: int = 0, startcol: int = 0,
                 engine: str | None = None, merge_cells: bool_t = True, encoding: lib.NoDefault = lib.no_default,
                 inf_rep: str = "inf", verbose: lib.NoDefault = lib.no_default,
                 freeze_panes: tuple[int, int] | None = None, storage_options: StorageOptions = None) -> None:
        super().to_excel(excel_writer, sheet_name, na_rep, float_format, columns, header, index, index_label, startrow,
                         startcol, engine, merge_cells, encoding, inf_rep, verbose, freeze_panes, storage_options)

    def to_json(self, path_or_buf: FilePath | WriteBuffer[bytes] | WriteBuffer[str] | None = None,
                orient: str | None = None, date_format: str | None = None, double_precision: int = 10,
                force_ascii: bool_t = True, date_unit: str = "ms",
                default_handler: Callable[[Any], JSONSerializable] | None = None, lines: bool_t = False,
                compression: CompressionOptions = "infer", index: bool_t = True, indent: int | None = None,
                storage_options: StorageOptions = None) -> str | None:
        return super().to_json(path_or_buf, orient, date_format, double_precision, force_ascii, date_unit,
                               default_handler, lines, compression, index, indent, storage_options)

    def to_hdf(self, path_or_buf: FilePath | HDFStore, key: str, mode: str = "a", complevel: int | None = None,
               complib: str | None = None, append: bool_t = False, format: str | None = None, index: bool_t = True,
               min_itemsize: int | dict[str, int] | None = None, nan_rep=None, dropna: bool_t | None = None,
               data_columns: Literal[True] | list[str] | None = None, errors: str = "strict",
               encoding: str = "UTF-8") -> None:
        super().to_hdf(path_or_buf, key, mode, complevel, complib, append, format, index, min_itemsize, nan_rep, dropna,
                       data_columns, errors, encoding)

    def to_sql(self, name: str, con, schema: str | None = None, if_exists: str = "fail", index: bool_t = True,
               index_label: IndexLabel = None, chunksize: int | None = None, dtype: DtypeArg | None = None,
               method: str | None = None) -> int | None:
        return super().to_sql(name, con, schema, if_exists, index, index_label, chunksize, dtype, method)

    def to_pickle(self, path: FilePath | WriteBuffer[bytes], compression: CompressionOptions = "infer",
                  protocol: int = pickle.HIGHEST_PROTOCOL, storage_options: StorageOptions = None) -> None:
        super().to_pickle(path, compression, protocol, storage_options)

    def to_clipboard(self, excel: bool_t = True, sep: str | None = None, **kwargs) -> None:
        super().to_clipboard(excel, sep, **kwargs)

    def to_xarray(self):
        return super().to_xarray()

    @overload
    def to_latex(
        self,
        buf: None = ...,
        columns: Sequence[Hashable] | None = ...,
        col_space: ColspaceArgType | None = ...,
        header: bool_t | Sequence[str] = ...,
        index: bool_t = ...,
        na_rep: str = ...,
        formatters: FormattersType | None = ...,
        float_format: FloatFormatType | None = ...,
        sparsify: bool_t | None = ...,
        index_names: bool_t = ...,
        bold_rows: bool_t = ...,
        column_format: str | None = ...,
        longtable: bool_t | None = ...,
        escape: bool_t | None = ...,
        encoding: str | None = ...,
        decimal: str = ...,
        multicolumn: bool_t | None = ...,
        multicolumn_format: str | None = ...,
        multirow: bool_t | None = ...,
        caption: str | tuple[str, str] | None = ...,
        label: str | None = ...,
        position: str | None = ...,
    ) -> str:
        ...

    @overload
    def to_latex(
        self,
        buf: FilePath | WriteBuffer[str],
        columns: Sequence[Hashable] | None = ...,
        col_space: ColspaceArgType | None = ...,
        header: bool_t | Sequence[str] = ...,
        index: bool_t = ...,
        na_rep: str = ...,
        formatters: FormattersType | None = ...,
        float_format: FloatFormatType | None = ...,
        sparsify: bool_t | None = ...,
        index_names: bool_t = ...,
        bold_rows: bool_t = ...,
        column_format: str | None = ...,
        longtable: bool_t | None = ...,
        escape: bool_t | None = ...,
        encoding: str | None = ...,
        decimal: str = ...,
        multicolumn: bool_t | None = ...,
        multicolumn_format: str | None = ...,
        multirow: bool_t | None = ...,
        caption: str | tuple[str, str] | None = ...,
        label: str | None = ...,
        position: str | None = ...,
    ) -> None:
        ...

    def to_latex(self, buf: FilePath | WriteBuffer[str] | None = None, columns: Sequence[Hashable] | None = None,
                 col_space: ColspaceArgType | None = None, header: bool_t | Sequence[str] = True, index: bool_t = True,
                 na_rep: str = "NaN", formatters: FormattersType | None = None,
                 float_format: FloatFormatType | None = None, sparsify: bool_t | None = None,
                 index_names: bool_t = True, bold_rows: bool_t = False, column_format: str | None = None,
                 longtable: bool_t | None = None, escape: bool_t | None = None, encoding: str | None = None,
                 decimal: str = ".", multicolumn: bool_t | None = None, multicolumn_format: str | None = None,
                 multirow: bool_t | None = None, caption: str | tuple[str, str] | None = None, label: str | None = None,
                 position: str | None = None) -> str | None:
        return super().to_latex(buf, columns, col_space, header, index, na_rep, formatters, float_format, sparsify,
                                index_names, bold_rows, column_format, longtable, escape, encoding, decimal,
                                multicolumn, multicolumn_format, multirow, caption, label, position)

    @overload
    def to_csv(
        self,
        path_or_buf: None = ...,
        sep: str = ...,
        na_rep: str = ...,
        float_format: str | Callable | None = ...,
        columns: Sequence[Hashable] | None = ...,
        header: bool_t | list[str] = ...,
        index: bool_t = ...,
        index_label: IndexLabel | None = ...,
        mode: str = ...,
        encoding: str | None = ...,
        compression: CompressionOptions = ...,
        quoting: int | None = ...,
        quotechar: str = ...,
        lineterminator: str | None = ...,
        chunksize: int | None = ...,
        date_format: str | None = ...,
        doublequote: bool_t = ...,
        escapechar: str | None = ...,
        decimal: str = ...,
        errors: str = ...,
        storage_options: StorageOptions = ...,
    ) -> str:
        ...

    @overload
    def to_csv(
        self,
        path_or_buf: FilePath | WriteBuffer[bytes] | WriteBuffer[str],
        sep: str = ...,
        na_rep: str = ...,
        float_format: str | Callable | None = ...,
        columns: Sequence[Hashable] | None = ...,
        header: bool_t | list[str] = ...,
        index: bool_t = ...,
        index_label: IndexLabel | None = ...,
        mode: str = ...,
        encoding: str | None = ...,
        compression: CompressionOptions = ...,
        quoting: int | None = ...,
        quotechar: str = ...,
        lineterminator: str | None = ...,
        chunksize: int | None = ...,
        date_format: str | None = ...,
        doublequote: bool_t = ...,
        escapechar: str | None = ...,
        decimal: str = ...,
        errors: str = ...,
        storage_options: StorageOptions = ...,
    ) -> None:
        ...

    def to_csv(self, path_or_buf: FilePath | WriteBuffer[bytes] | WriteBuffer[str] | None = None, sep: str = ",",
               na_rep: str = "", float_format: str | Callable | None = None, columns: Sequence[Hashable] | None = None,
               header: bool_t | list[str] = True, index: bool_t = True, index_label: IndexLabel | None = None,
               mode: str = "w", encoding: str | None = None, compression: CompressionOptions = "infer",
               quoting: int | None = None, quotechar: str = '"', lineterminator: str | None = None,
               chunksize: int | None = None, date_format: str | None = None, doublequote: bool_t = True,
               escapechar: str | None = None, decimal: str = ".", errors: str = "strict",
               storage_options: StorageOptions = None) -> str | None:
        return super().to_csv(path_or_buf, sep, na_rep, float_format, columns, header, index, index_label, mode,
                              encoding, compression, quoting, quotechar, lineterminator, chunksize, date_format,
                              doublequote, escapechar, decimal, errors, storage_options)

    def _take(self: NDFrameT, indices, axis=0, convert_indices: bool_t = True) -> NDFrameT:
        return super()._take(indices, axis, convert_indices)

    def xs(self: NDFrameT, key: IndexLabel, axis: Axis = 0, level: IndexLabel = None,
           drop_level: bool_t = True) -> NDFrameT:
        return super().xs(key, axis, level, drop_level)

    def _set_is_copy(self, ref: NDFrame, copy: bool_t = True) -> None:
        super()._set_is_copy(ref, copy)

    def _check_setitem_copy(self, t="setting", force=False):
        super()._check_setitem_copy(t, force)

    def __delitem__(self, key) -> None:
        super().__delitem__(key)

    def _check_inplace_and_allows_duplicate_labels(self, inplace):
        return super()._check_inplace_and_allows_duplicate_labels(inplace)

    def get(self, key, default=None):
        return super().get(key, default)

    @property
    def _is_view(self) -> bool_t:
        return super()._is_view

    def reindex_like(self: NDFrameT, other, method: str | None = None, copy: bool_t = True, limit=None,
                     tolerance=None) -> NDFrameT:
        return super().reindex_like(other, method, copy, limit, tolerance)

    def _drop_axis(self: NDFrameT, labels, axis, level=None, errors: IgnoreRaise = "raise",
                   only_slice: bool_t = False) -> NDFrameT:
        return super()._drop_axis(labels, axis, level, errors, only_slice)

    def _update_inplace(self, result, verify_is_copy: bool_t = True) -> None:
        super()._update_inplace(result, verify_is_copy)

    def add_prefix(self: NDFrameT, prefix: str) -> NDFrameT:
        return super().add_prefix(prefix)

    def add_suffix(self: NDFrameT, suffix: str) -> NDFrameT:
        return super().add_suffix(suffix)

    def _reindex_axes(self: NDFrameT, axes, level, limit, tolerance, method, fill_value, copy) -> NDFrameT:
        return super()._reindex_axes(axes, level, limit, tolerance, method, fill_value, copy)

    def _reindex_multi(self, axes, copy, fill_value):
        return super()._reindex_multi(axes, copy, fill_value)

    def _reindex_with_indexers(self: NDFrameT, reindexers, fill_value=None, copy: bool_t = False,
                               allow_dups: bool_t = False) -> NDFrameT:
        return super()._reindex_with_indexers(reindexers, fill_value, copy, allow_dups)

    def filter(self: NDFrameT, items=None, like: str | None = None, regex: str | None = None, axis=None) -> NDFrameT:
        return super().filter(items, like, regex, axis)

    def head(self: NDFrameT, n: int = 5) -> NDFrameT:
        return super().head(n)

    def tail(self: NDFrameT, n: int = 5) -> NDFrameT:
        return super().tail(n)

    def sample(self: NDFrameT, n: int | None = None, frac: float | None = None, replace: bool_t = False, weights=None,
               random_state: RandomState | None = None, axis: Axis | None = None,
               ignore_index: bool_t = False) -> NDFrameT:
        return super().sample(n, frac, replace, weights, random_state, axis, ignore_index)

    def pipe(self, func: Callable[..., T] | tuple[Callable[..., T], str], *args, **kwargs) -> T:
        return super().pipe(func, *args, **kwargs)

    def __finalize__(self: NDFrameT, other, method: str | None = None, **kwargs) -> NDFrameT:
        return super().__finalize__(other, method, **kwargs)

    def __getattr__(self, name: str):
        return super().__getattr__(name)

    def __setattr__(self, name: str, value) -> None:
        super().__setattr__(name, value)

    def _dir_additions(self) -> set[str]:
        return super()._dir_additions()

    def _protect_consolidate(self, f):
        return super()._protect_consolidate(f)

    def _consolidate_inplace(self) -> None:
        super()._consolidate_inplace()

    def _consolidate(self):
        return super()._consolidate()

    def _check_inplace_setting(self, value) -> bool_t:
        return super()._check_inplace_setting(value)

    def _get_numeric_data(self: NDFrameT) -> NDFrameT:
        return super()._get_numeric_data()

    def _get_bool_data(self):
        return super()._get_bool_data()

    def astype(self: NDFrameT, dtype, copy: bool_t = True, errors: IgnoreRaise = "raise") -> NDFrameT:
        return super().astype(dtype, copy, errors)

    def copy(self: NDFrameT, deep: bool_t | None = True) -> NDFrameT:
        return super().copy(deep)

    def __copy__(self: NDFrameT, deep: bool_t = True) -> NDFrameT:
        return super().__copy__(deep)

    def __deepcopy__(self: NDFrameT, memo=None) -> NDFrameT:
        return super().__deepcopy__(memo)

    def _convert(self: NDFrameT, datetime: bool_t = False, numeric: bool_t = False,
                 timedelta: bool_t = False) -> NDFrameT:
        return super()._convert(datetime, numeric, timedelta)

    def infer_objects(self: NDFrameT) -> NDFrameT:
        return super().infer_objects()

    def convert_dtypes(self: NDFrameT, infer_objects: bool_t = True, convert_string: bool_t = True,
                       convert_integer: bool_t = True, convert_boolean: bool_t = True,
                       convert_floating: bool_t = True) -> NDFrameT:
        return super().convert_dtypes(infer_objects, convert_string, convert_integer, convert_boolean, convert_floating)

    def asof(self, where, subset=None):
        return super().asof(where, subset)

    def _clip_with_scalar(self, lower, upper, inplace: bool_t = False):
        return super()._clip_with_scalar(lower, upper, inplace)

    def _clip_with_one_bound(self, threshold, method, axis, inplace):
        return super()._clip_with_one_bound(threshold, method, axis, inplace)

    def at_time(self: NDFrameT, time, asof: bool_t = False, axis=None) -> NDFrameT:
        return super().at_time(time, asof, axis)

    def between_time(self: NDFrameT, start_time, end_time, include_start: bool_t | lib.NoDefault = lib.no_default,
                     include_end: bool_t | lib.NoDefault = lib.no_default, inclusive: IntervalClosedType | None = None,
                     axis=None) -> NDFrameT:
        return super().between_time(start_time, end_time, include_start, include_end, inclusive, axis)

    def first(self: NDFrameT, offset) -> NDFrameT:
        return super().first(offset)

    def last(self: NDFrameT, offset) -> NDFrameT:
        return super().last(offset)

    def rank(self: NDFrameT, axis=0, method: str = "average",
             numeric_only: bool_t | None | lib.NoDefault = lib.no_default, na_option: str = "keep",
             ascending: bool_t = True, pct: bool_t = False) -> NDFrameT:
        return super().rank(axis, method, numeric_only, na_option, ascending, pct)

    def _align_frame(self, other, join="outer", axis=None, level=None, copy: bool_t = True, fill_value=None,
                     method=None, limit=None, fill_axis=0):
        return super()._align_frame(other, join, axis, level, copy, fill_value, method, limit, fill_axis)

    def _align_series(self, other, join="outer", axis=None, level=None, copy: bool_t = True, fill_value=None,
                      method=None, limit=None, fill_axis=0):
        return super()._align_series(other, join, axis, level, copy, fill_value, method, limit, fill_axis)

    def _where(self, cond, other=lib.no_default, inplace=False, axis=None, level=None):
        return super()._where(cond, other, inplace, axis, level)

    def slice_shift(self: NDFrameT, periods: int = 1, axis=0) -> NDFrameT:
        return super().slice_shift(periods, axis)

    def tshift(self: NDFrameT, periods: int = 1, freq=None, axis: Axis = 0) -> NDFrameT:
        return super().tshift(periods, freq, axis)

    def truncate(self: NDFrameT, before=None, after=None, axis=None, copy: bool_t = True) -> NDFrameT:
        return super().truncate(before, after, axis, copy)

    def tz_convert(self: NDFrameT, tz, axis=0, level=None, copy: bool_t = True) -> NDFrameT:
        return super().tz_convert(tz, axis, level, copy)

    def tz_localize(self: NDFrameT, tz, axis=0, level=None, copy: bool_t = True, ambiguous="raise",
                    nonexistent: str = "raise") -> NDFrameT:
        return super().tz_localize(tz, axis, level, copy, ambiguous, nonexistent)

    def describe(self: NDFrameT, percentiles=None, include=None, exclude=None,
                 datetime_is_numeric: bool_t = False) -> NDFrameT:
        return super().describe(percentiles, include, exclude, datetime_is_numeric)

    def pct_change(self: NDFrameT, periods=1, fill_method="pad", limit=None, freq=None, **kwargs) -> NDFrameT:
        return super().pct_change(periods, fill_method, limit, freq, **kwargs)

    def _agg_by_level(self, name: str, axis: Axis = 0, level: Level = 0, skipna: bool_t = True, **kwargs):
        return super()._agg_by_level(name, axis, level, skipna, **kwargs)

    def _logical_func(self, name: str, func, axis: Axis = 0, bool_only: bool_t | None = None, skipna: bool_t = True,
                      level: Level | None = None, **kwargs) -> Series | bool_t:
        return super()._logical_func(name, func, axis, bool_only, skipna, level, **kwargs)

    def all(self, axis: Axis = 0, bool_only: bool_t | None = None, skipna: bool_t = True, level: Level | None = None,
            **kwargs) -> Series | bool_t:
        return super().all(axis, bool_only, skipna, level, **kwargs)

    def _accum_func(self, name: str, func, axis: Axis | None = None, skipna: bool_t = True, *args, **kwargs):
        return super()._accum_func(name, func, axis, skipna, *args, **kwargs)

    def cummax(self, axis: Axis | None = None, skipna: bool_t = True, *args, **kwargs):
        return super().cummax(axis, skipna, *args, **kwargs)

    def cummin(self, axis: Axis | None = None, skipna: bool_t = True, *args, **kwargs):
        return super().cummin(axis, skipna, *args, **kwargs)

    def cumsum(self, axis: Axis | None = None, skipna: bool_t = True, *args, **kwargs):
        return super().cumsum(axis, skipna, *args, **kwargs)

    def cumprod(self, axis: Axis | None = None, skipna: bool_t = True, *args, **kwargs):
        return super().cumprod(axis, skipna, *args, **kwargs)

    def _stat_function_ddof(self, name: str, func, axis: Axis | None = None, skipna: bool_t = True,
                            level: Level | None = None, ddof: int = 1, numeric_only: bool_t | None = None,
                            **kwargs) -> Series | float:
        return super()._stat_function_ddof(name, func, axis, skipna, level, ddof, numeric_only, **kwargs)

    def sem(self, axis: Axis | None = None, skipna: bool_t = True, level: Level | None = None, ddof: int = 1,
            numeric_only: bool_t | None = None, **kwargs) -> Series | float:
        return super().sem(axis, skipna, level, ddof, numeric_only, **kwargs)

    def var(self, axis: Axis | None = None, skipna: bool_t = True, level: Level | None = None, ddof: int = 1,
            numeric_only: bool_t | None = None, **kwargs) -> Series | float:
        return super().var(axis, skipna, level, ddof, numeric_only, **kwargs)

    def std(self, axis: Axis | None = None, skipna: bool_t = True, level: Level | None = None, ddof: int = 1,
            numeric_only: bool_t | None = None, **kwargs) -> Series | float:
        return super().std(axis, skipna, level, ddof, numeric_only, **kwargs)

    def _stat_function(self, name: str, func, axis: Axis | None | lib.NoDefault = None, skipna: bool_t = True,
                       level: Level | None = None, numeric_only: bool_t | None = None, **kwargs):
        return super()._stat_function(name, func, axis, skipna, level, numeric_only, **kwargs)

    def mean(self, axis: Axis | None | lib.NoDefault = lib.no_default, skipna: bool_t = True,
             level: Level | None = None, numeric_only: bool_t | None = None, **kwargs) -> Series | float:
        return super().mean(axis, skipna, level, numeric_only, **kwargs)

    def median(self, axis: Axis | None | lib.NoDefault = lib.no_default, skipna: bool_t = True,
               level: Level | None = None, numeric_only: bool_t | None = None, **kwargs) -> Series | float:
        return super().median(axis, skipna, level, numeric_only, **kwargs)

    def skew(self, axis: Axis | None | lib.NoDefault = lib.no_default, skipna: bool_t = True,
             level: Level | None = None, numeric_only: bool_t | None = None, **kwargs) -> Series | float:
        return super().skew(axis, skipna, level, numeric_only, **kwargs)

    def kurt(self, axis: Axis | None | lib.NoDefault = lib.no_default, skipna: bool_t = True,
             level: Level | None = None, numeric_only: bool_t | None = None, **kwargs) -> Series | float:
        return super().kurt(axis, skipna, level, numeric_only, **kwargs)

    def _min_count_stat_function(self, name: str, func, axis: Axis | None = None, skipna: bool_t = True,
                                 level: Level | None = None, numeric_only: bool_t | None = None, min_count: int = 0,
                                 **kwargs):
        return super()._min_count_stat_function(name, func, axis, skipna, level, numeric_only, min_count, **kwargs)

    def sum(self, axis: Axis | None = None, skipna: bool_t = True, level: Level | None = None,
            numeric_only: bool_t | None = None, min_count=0, **kwargs):
        return super().sum(axis, skipna, level, numeric_only, min_count, **kwargs)

    def prod(self, axis: Axis | None = None, skipna: bool_t = True, level: Level | None = None,
             numeric_only: bool_t | None = None, min_count: int = 0, **kwargs):
        return super().prod(axis, skipna, level, numeric_only, min_count, **kwargs)

    def mad(self, axis: Axis | None = None, skipna: bool_t = True, level: Level | None = None) -> Series | float:
        return super().mad(axis, skipna, level)

    @classmethod
    def _add_numeric_operations(cls):
        super()._add_numeric_operations()

    def rolling(self, window: int | timedelta | BaseOffset | BaseIndexer, min_periods: int | None = None,
                center: bool_t = False, win_type: str | None = None, on: str | None = None, axis: Axis = 0,
                closed: str | None = None, step: int | None = None, method: str = "single") -> Window | Rolling:
        return super().rolling(window, min_periods, center, win_type, on, axis, closed, step, method)

    def expanding(self, min_periods: int = 1, center: bool_t | None = None, axis: Axis = 0,
                  method: str = "single") -> Expanding:
        return super().expanding(min_periods, center, axis, method)

    def ewm(self, com: float | None = None, span: float | None = None,
            halflife: float | TimedeltaConvertibleTypes | None = None, alpha: float | None = None,
            min_periods: int | None = 0, adjust: bool_t = True, ignore_na: bool_t = False, axis: Axis = 0,
            times: str | np.ndarray | DataFrame | Series | None = None,
            method: str = "single") -> ExponentialMovingWindow:
        return super().ewm(com, span, halflife, alpha, min_periods, adjust, ignore_na, axis, times, method)

    def _inplace_method(self, other, op):
        return super()._inplace_method(other, op)

    def __iadd__(self: NDFrameT, other) -> NDFrameT:
        return super().__iadd__(other)

    def __isub__(self: NDFrameT, other) -> NDFrameT:
        return super().__isub__(other)

    def __imul__(self: NDFrameT, other) -> NDFrameT:
        return super().__imul__(other)

    def __itruediv__(self: NDFrameT, other) -> NDFrameT:
        return super().__itruediv__(other)

    def __ifloordiv__(self: NDFrameT, other) -> NDFrameT:
        return super().__ifloordiv__(other)

    def __imod__(self: NDFrameT, other) -> NDFrameT:
        return super().__imod__(other)

    def __ipow__(self: NDFrameT, other) -> NDFrameT:
        return super().__ipow__(other)

    def __iand__(self: NDFrameT, other) -> NDFrameT:
        return super().__iand__(other)

    def __ior__(self: NDFrameT, other) -> NDFrameT:
        return super().__ior__(other)

    def __ixor__(self: NDFrameT, other) -> NDFrameT:
        return super().__ixor__(other)

    def _find_valid_index(self, *, how: str) -> Hashable | None:
        return super()._find_valid_index(how=how)

    def first_valid_index(self) -> Hashable | None:
        return super().first_valid_index()

    def last_valid_index(self) -> Hashable | None:
        return super().last_valid_index()

    def _reset_cache(self, key: str | None = None) -> None:
        super()._reset_cache(key)

    def __sizeof__(self) -> int:
        return super().__sizeof__()

    def _dir_deletions(self) -> set[str]:
        return super()._dir_deletions()

    def __dir__(self) -> list[str]:
        return super().__dir__()

    @property
    def iloc(self) -> _iLocIndexer:
        return super().iloc

    @property
    def loc(self) -> _LocIndexer:
        return super().loc

    @property
    def at(self) -> _AtIndexer:
        return super().at

    @property
    def iat(self) -> _iAtIndexer:
        return super().iat

    @property
    def __class__(self: _T) -> Type[_T]:
        return super().__class__

    def __new__(cls: Type[_T]) -> _T:
        return super().__new__(cls)

    def __str__(self) -> str:
        return super().__str__()

    def __hash__(self) -> int:
        return super().__hash__()

    def __format__(self, format_spec: str) -> str:
        return super().__format__(format_spec)

    def __getattribute__(self, name: str) -> Any:
        return super().__getattribute__(name)

    def __delattr__(self, name: str) -> None:
        super().__delattr__(name)

    def __reduce__(self) -> str | Tuple[Any, ...]:
        return super().__reduce__()

    def __reduce_ex__(self, protocol: SupportsIndex) -> str | Tuple[Any, ...]:
        return super().__reduce_ex__(protocol)

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()