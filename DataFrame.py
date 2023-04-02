import datetime
import pickle
from datetime import timedelta
from typing import Tuple, Any, Hashable, Callable, Sequence, Literal, NoReturn, Mapping, Iterable, \
    Iterator
from typing_extensions import SupportsIndex

from collections import namedtuple
import numpy as np
import pandas as pd
from pandas import Series, HDFStore, Index, MultiIndex, Flags, get_option
from pandas._libs import lib
from pandas._libs.lib import NoDefault, no_default
from pandas._libs.tslibs import BaseOffset
from pandas._typing import NDFrameT, TimedeltaConvertibleTypes, Axis, Level, IntervalClosedType, IgnoreRaise, \
    RandomState, IndexLabel, FilePath, WriteBuffer, CompressionOptions, StorageOptions, ColspaceArgType, FormattersType, \
    FloatFormatType, DtypeArg, JSONSerializable, npt, ArrayLike, Renamer, AnyArrayLike, DtypeObj, Manager, Dtype, \
    Frequency, TimestampConvertibleTypes, FillnaOptions, QuantileInterpolation, Suffixes, PythonFuncType, AggFuncType, \
    SortKind, NaPosition, IndexKeyFunc, ValueKeyFunc, Scalar, ReadBuffer, Axes
from pandas.core.arrays import DatetimeArray, TimedeltaArray, PeriodArray
from pandas.core.generic import bool_t, NDFrame
from pandas.core.indexers.objects import BaseIndexer
from pandas.core.indexing import _iAtIndexer, _AtIndexer, _LocIndexer, _iLocIndexer
from pandas.core.internals import SingleDataManager
from pandas.core.resample import Resampler
from pandas.core.window import ExponentialMovingWindow, Expanding, Window, Rolling
from pandas.io.formats import format as fmt
from pandas.io.formats.style import Styler

import datatable as dt
from pandas.util import Appender


def _to_pandas(table):
    d = table.to_pandas()
    if 'index' in d.columns:
        d.set_index(d, inplace=True, drop=True)
    return d


class DataFrame(pd.DataFrame):
    @property
    def _constructor(self) -> Callable[..., pd.DataFrame]:
        return dt.Frame()

    def __init__(self, data=None, index: Axes | None = None, columns: Axes | None = None, dtype: Dtype | None = None,
                 copy: bool | None = None) -> None:
        if isinstance(data, DataFrame):
            self._frame = dt.Frame(data)
        else:
            self._frame = dt.Frame(data, names=columns, type=dtype)
            if index is not None:
                self._frame['index'] = dt.Frame(index)

    def __dataframe__(self, nan_as_null: bool = False, allow_copy: bool = True) -> pd.DataFrameXchg:
        return _to_pandas(self._frame).__dataframe__(nan_as_null, allow_copy)

    @property
    def axes(self) -> list[Index]:
        colnames = list(self._frame.names)
        if 'index' in colnames:
            colnames.remove('index')
            index = self._frame['indexx']
        else:
            index = range(self._frame.shape[0])
        return [pd.Index(index), pd.Index(colnames)]

    @property
    def shape(self) -> tuple[int, int]:
        return self._frame.shape

    @property
    def _is_homogeneous_type(self) -> bool:
        return len(set(self._frame.types)) == 1

    @property
    def _can_fast_transpose(self) -> bool:
        return _to_pandas(self._frame)._can_fast_transpose

    @property
    def _values(self) -> np.ndarray | DatetimeArray | TimedeltaArray | PeriodArray:
        return self._frame.to_numpy()

    def _repr_fits_vertical_(self) -> bool:
        return self._frame.shape[0] <= dt.options.display.max_nrows

    def _repr_fits_horizontal_(self, ignore_width: bool = False) -> bool:
        return self._frame.shape[1] <= dt.options.display.max_column_width

    def _info_repr(self) -> bool:
        return _to_pandas(self._frame)._info_repr()

    def __repr__(self) -> str:
        return _to_pandas(self._frame).__repr__()

    def _repr_html_(self) -> str | None:
        return _to_pandas(self._frame).__repr__()

    def to_string(self, buf: FilePath | WriteBuffer[str] | None = None, columns: Sequence[str] | None = None,
                  col_space: int | list[int] | dict[Hashable, int] | None = None, header: bool | Sequence[str] = True,
                  index: bool = True, na_rep: str = "NaN", formatters: fmt.FormattersType | None = None,
                  float_format: fmt.FloatFormatType | None = None, sparsify: bool | None = None,
                  index_names: bool = True, justify: str | None = None, max_rows: int | None = None,
                  max_cols: int | None = None, show_dimensions: bool = False, decimal: str = ".",
                  line_width: int | None = None, min_rows: int | None = None, max_colwidth: int | None = None,
                  encoding: str | None = None) -> str | None:
        return self._frame.__str__()

    @property
    def style(self) -> Styler:
        return _to_pandas(self._frame).style()

    @Appender
    def items(self) -> Iterable[tuple[Hashable, Series]]:
        for c in self._frame:
            yield c.names[0], c.to_pandas()._series

    @Appender
    def iteritems(self) -> Iterable[tuple[Hashable, Series]]:
        yield from self.items()

    def iterrows(self) -> Iterable[tuple[Hashable, Series]]:
        if 'index' in self._frame.names:
            return [(i, self._frame[dt.f.index == i, :].to_pandas()._series) for i in self._frame['index'].to_list()]
        else:
            return [(i, self._frame[dt.f.index == i, :].to_pandas()._series) for i in range(self._frame.shape[0])]

    def itertuples(self, index: bool = True, name: str | None = "Pandas") -> Iterable[tuple[Any, ...]]:
        nt = namedtuple('pandas', self._frame.names)
        return [nt(*t) for t in self._frame.to_tuples()]

    def __len__(self) -> int:
        return self._frame.nrows

    def dot(self, other: AnyArrayLike):
        return DataFrame(np.dot(self._frame, other))

    def __matmul__(self, other: AnyArrayLike | pd.DataFrame) -> pd.DataFrame | Series:
        return DataFrame(np.matmul(self._frame, other))

    def __rmatmul__(self, other) -> pd.DataFrame:
        return DataFrame(self._frame @ other)

    @classmethod
    def from_dict(cls, data: dict, orient: str = "columns", dtype: Dtype | None = None,
                  columns: Axes | None = None) -> pd.DataFrame:
        return DataFrame(data)

    def to_numpy(self, dtype: npt.DTypeLike | None = None, copy: bool = False,
                 na_value: object = lib.no_default) -> np.ndarray:
        return self._frame.to_numpy()

    def to_dict(self, orient: Literal[
        "dict", "list", "series", "split", "tight", "records", "index"
    ] = "dict", into: type[dict] = dict) -> dict | list[dict]:
        return self._frame.to_dict()

    def to_gbq(self, destination_table: str, project_id: str | None = None, chunksize: int | None = None,
               reauth: bool = False, if_exists: str = "fail", auth_local_webserver: bool = True,
               table_schema: list[dict[str, str]] | None = None, location: str | None = None, progress_bar: bool = True,
               credentials=None) -> None:
        _to_pandas(self._frame).to_gbq(destination_table, project_id, chunksize, reauth, if_exists,
                                       auth_local_webserver, table_schema,
                                       location, progress_bar, credentials)

    @classmethod
    def from_records(cls, data, index=None, exclude=None, columns=None, coerce_float: bool = False,
                     nrows: int | None = None) -> pd.DataFrame:
        return DataFrame(pd.DataFrame.from_records(cls, data, index, exclude, columns, coerce_float, nrows))

    def to_records(self, index: bool = True, column_dtypes=None, index_dtypes=None) -> np.recarray:
        return _to_pandas(self._frame).to_records(index, column_dtypes, index_dtypes)

    @classmethod
    def _from_arrays(cls, arrays, columns, index, dtype: Dtype | None = None,
                     verify_integrity: bool = True) -> pd.DataFrame:
        return DataFrame(arrays, columns, index, dtype, verify_integrity)

    def to_stata(self, path: FilePath | WriteBuffer[bytes], convert_dates: dict[Hashable, str] | None = None,
                 write_index: bool = True, byteorder: str | None = None, time_stamp: datetime.datetime | None = None,
                 data_label: str | None = None, variable_labels: dict[Hashable, str] | None = None,
                 version: int | None = 114, convert_strl: Sequence[Hashable] | None = None,
                 compression: CompressionOptions = "infer", storage_options: StorageOptions = None, *,
                 value_labels: dict[Hashable, dict[float, str]] | None = None) -> None:
        _to_pandas(self._frame).to_stata(path, convert_dates, write_index, byteorder, time_stamp, data_label,
                                         variable_labels, version,
                                         convert_strl, compression, storage_options, value_labels=value_labels)

    def to_feather(self, path: FilePath | WriteBuffer[bytes], **kwargs) -> None:
        _to_pandas(self._frame).to_feather(path, **kwargs)

    def to_markdown(self, buf: FilePath | WriteBuffer[str] | None = None, mode: str = "wt", index: bool = True,
                    storage_options: StorageOptions = None, **kwargs) -> str | None:
        return _to_pandas(self._frame).to_markdown(buf, mode, index, storage_options, **kwargs)

    def to_parquet(self, path: FilePath | WriteBuffer[bytes] | None = None, engine: str = "auto",
                   compression: str | None = "snappy", index: bool | None = None,
                   partition_cols: list[str] | None = None, storage_options: StorageOptions = None,
                   **kwargs) -> bytes | None:
        return _to_pandas(self._frame).to_parquet(path, engine, compression, index, partition_cols, storage_options,
                                                  **kwargs)

    def to_orc(self, path: FilePath | WriteBuffer[bytes] | None = None, *, engine: Literal["pyarrow"] = "pyarrow",
               index: bool | None = None, engine_kwargs: dict[str, Any] | None = None) -> bytes | None:
        return _to_pandas(self._frame).to_orc(path, engine=engine, index=index, engine_kwargs=engine_kwargs)

    def to_html(self, buf: FilePath | WriteBuffer[str] | None = None, columns: Sequence[Level] | None = None,
                col_space: ColspaceArgType | None = None, header: bool | Sequence[str] = True, index: bool = True,
                na_rep: str = "NaN", formatters: FormattersType | None = None,
                float_format: FloatFormatType | None = None, sparsify: bool | None = None, index_names: bool = True,
                justify: str | None = None, max_rows: int | None = None, max_cols: int | None = None,
                show_dimensions: bool | str = False, decimal: str = ".", bold_rows: bool = True,
                classes: str | list | tuple | None = None, escape: bool = True, notebook: bool = False,
                border: int | bool | None = None, table_id: str | None = None, render_links: bool = False,
                encoding: str | None = None) -> str | None:
        return _to_pandas(self._frame).to_html(buf, columns, col_space, header, index, na_rep, formatters, float_format,
                                               sparsify,
                                               index_names, justify, max_rows, max_cols, show_dimensions, decimal,
                                               bold_rows, classes,
                                               escape, notebook, border, table_id, render_links, encoding)

    def to_xml(self, path_or_buffer: FilePath | WriteBuffer[bytes] | WriteBuffer[str] | None = None, index: bool = True,
               root_name: str | None = "data", row_name: str | None = "row", na_rep: str | None = None,
               attr_cols: list[str] | None = None, elem_cols: list[str] | None = None,
               namespaces: dict[str | None, str] | None = None, prefix: str | None = None, encoding: str = "utf-8",
               xml_declaration: bool | None = True, pretty_print: bool | None = True, parser: str | None = "lxml",
               stylesheet: FilePath | ReadBuffer[str] | ReadBuffer[bytes] | None = None,
               compression: CompressionOptions = "infer", storage_options: StorageOptions = None) -> str | None:
        return _to_pandas(self._frame).to_xml(path_or_buffer, index, root_name, row_name, na_rep, attr_cols, elem_cols,
                                              namespaces,
                                              prefix, encoding, xml_declaration, pretty_print, parser, stylesheet,
                                              compression,
                                              storage_options)

    def info(self, verbose: bool | None = None, buf: WriteBuffer[str] | None = None, max_cols: int | None = None,
             memory_usage: bool | str | None = None, show_counts: bool | None = None,
             null_counts: bool | None = None) -> None:
        _to_pandas(self._frame).info(verbose, buf, max_cols, memory_usage, show_counts, null_counts)

    def memory_usage(self, index: bool = True, deep: bool = False) -> Series:
        return self._frame.__sizeof__()

    def transpose(self, *args, copy: bool = False) -> pd.DataFrame:
        new = DataFrame(np.transpose(self._frame))
        new['index'] = self._frame.names
        if not copy:
            self._frame = new
        else:
            return new

    @property
    def T(self) -> pd.DataFrame:
        new = DataFrame(np.transpose(self._frame))
        new['index'] = self._frame.names
        return new

    def _ixs(self, i: int, axis: int = 0) -> Series:
        # TODO
        if axis == 0:
            return self._frame[i, :].to_pandas()._series
        else:
            return self._frame[:, i].to_pandas()._series

    def _get_column_array(self, i: int) -> ArrayLike:
        return self._frame[i].to_numpy()

    def _iter_column_arrays(self) -> Iterator[ArrayLike]:
        for i in range(len(self.columns)):
            yield self._get_column_array(i)

    def __getitem__(self, key):
        return _to_pandas(self._frame).__getitem__(key)

    def _getitem_bool_array(self, key):
        return _to_pandas(self._frame)._getitem_bool_array(key)

    def _getitem_multilevel(self, key):
        return _to_pandas(self._frame)._getitem_multilevel(key)

    def _get_value(self, index, col, takeable: bool = False) -> Scalar:
        return _to_pandas(self._frame)._get_value(index, col, takeable)

    def isetitem(self, loc, value) -> None:
        _to_pandas(self._frame).isetitem(loc, value)

    def __setitem__(self, key, value):
        _to_pandas(self._frame).__setitem__(key, value)

    def _setitem_slice(self, key: slice, value):
        _to_pandas(self._frame)._setitem_slice(key, value)

    def _setitem_array(self, key, value):
        return _to_pandas(self._frame)._setitem_array(key, value)

    def _iset_not_inplace(self, key, value):
        return _to_pandas(self._frame)._iset_not_inplace(key, value)

    def _setitem_frame(self, key, value):
        return _to_pandas(self._frame)._setitem_frame(key, value)

    def _set_item_frame_value(self, key, value: pd.DataFrame) -> None:
        _to_pandas(self._frame)._set_item_frame_value(key, value)

    def _iset_item_mgr(self, loc: int | slice | np.ndarray, value, inplace: bool = False) -> None:
        _to_pandas(self._frame)._iset_item_mgr(loc, value, inplace)

    def _set_item_mgr(self, key, value: ArrayLike) -> None:
        _to_pandas(self._frame)._set_item_mgr(key, value)

    def _iset_item(self, loc: int, value) -> None:
        _to_pandas(self._frame)._iset_item(loc, value)

    def _set_item(self, key, value) -> None:
        _to_pandas(self._frame)._set_item(key, value)

    def _set_value(self, index: IndexLabel, col, value: Scalar, takeable: bool = False) -> None:
        _to_pandas(self._frame)._set_value(index, col, value, takeable)

    def _ensure_valid_index(self, value) -> None:
        _to_pandas(self._frame)._ensure_valid_index(value)

    def _box_col_values(self, values: SingleDataManager, loc: int) -> Series:
        return _to_pandas(self._frame)._box_col_values(values, loc)

    def _clear_item_cache(self) -> None:
        _to_pandas(self._frame)._clear_item_cache()

    def _get_item_cache(self, item: Hashable) -> Series:
        return _to_pandas(self._frame)._get_item_cache(item)

    def _reset_cacher(self) -> None:
        _to_pandas(self._frame)._reset_cacher()

    def _maybe_cache_changed(self, item, value: Series, inplace: bool) -> None:
        _to_pandas(self._frame)._maybe_cache_changed(item, value, inplace)

    def query(self, expr: str, inplace: bool = False, **kwargs) -> pd.DataFrame | None:
        return _to_pandas(self._frame).query(expr, inplace, **kwargs)

    def eval(self, expr: str, inplace: bool = False, **kwargs) -> Any | None:
        return _to_pandas(self._frame).eval(expr, inplace, **kwargs)

    def select_dtypes(self, include=None, exclude=None) -> pd.DataFrame:
        return _to_pandas(self._frame).select_dtypes(include, exclude)

    def insert(self, loc: int, column: Hashable, value: Scalar | AnyArrayLike,
               allow_duplicates: bool | lib.NoDefault = lib.no_default) -> None:
        _to_pandas(self._frame).insert(loc, column, value, allow_duplicates)

    def assign(self, **kwargs) -> pd.DataFrame:
        return _to_pandas(self._frame).assign(**kwargs)

    def _sanitize_column(self, value) -> ArrayLike:
        return _to_pandas(self._frame)._sanitize_column(value)

    @property
    def _series(self):
        return _to_pandas(self._frame)._series

    def lookup(self, row_labels: Sequence[IndexLabel], col_labels: Sequence[IndexLabel]) -> np.ndarray:
        return _to_pandas(self._frame).lookup(row_labels, col_labels)

    def _reindex_axes(self, axes, level, limit, tolerance, method, fill_value, copy):
        return _to_pandas(self._frame)._reindex_axes(axes, level, limit, tolerance, method, fill_value, copy)

    def _reindex_index(self, new_index, method, copy: bool, level: Level, fill_value=np.nan, limit=None,
                       tolerance=None):
        return _to_pandas(self._frame)._reindex_index(new_index, method, copy, level, fill_value, limit, tolerance)

    def _reindex_columns(self, new_columns, method, copy: bool, level: Level, fill_value=None, limit=None,
                         tolerance=None):
        return _to_pandas(self._frame)._reindex_columns(new_columns, method, copy, level, fill_value, limit, tolerance)

    def _reindex_multi(self, axes: dict[str, Index], copy: bool, fill_value) -> pd.DataFrame:
        return _to_pandas(self._frame)._reindex_multi(axes, copy, fill_value)

    def align(self, other: pd.DataFrame, join: Literal["outer", "inner", "left", "right"] = "outer",
              axis: Axis | None = None, level: Level = None, copy: bool = True, fill_value=None,
              method: FillnaOptions | None = None, limit: int | None = None, fill_axis: Axis = 0,
              broadcast_axis: Axis | None = None) -> pd.DataFrame:
        return _to_pandas(self._frame).align(other, join, axis, level, copy, fill_value, method, limit, fill_axis,
                                             broadcast_axis)

    def set_axis(self, labels, axis: Axis = 0, inplace: bool | lib.NoDefault = lib.no_default, *,
                 copy: bool | lib.NoDefault = lib.no_default):
        return _to_pandas(self._frame).set_axis(labels, axis, inplace, copy=copy)

    def reindex(self, *args, **kwargs) -> pd.DataFrame:
        return _to_pandas(self._frame).reindex(*args, **kwargs)

    def drop(self, labels: IndexLabel = None, axis: Axis = 0, index: IndexLabel = None, columns: IndexLabel = None,
             level: Level = None, inplace: bool = False, errors: IgnoreRaise = "raise") -> pd.DataFrame | None:
        return _to_pandas(self._frame).drop(labels, axis, index, columns, level, inplace, errors)

    def rename(self, mapper: Renamer | None = None, *, index: Renamer | None = None, columns: Renamer | None = None,
               axis: Axis | None = None, copy: bool | None = None, inplace: bool = False, level: Level = None,
               errors: IgnoreRaise = "ignore") -> pd.DataFrame | None:
        return _to_pandas(self._frame).rename(mapper, index=index, columns=columns, axis=axis, copy=copy,
                                              inplace=inplace, level=level,
                                              errors=errors)

    def fillna(self, value: Hashable | Mapping | Series | pd.DataFrame = None, method: FillnaOptions | None = None,
               axis: Axis | None = None, inplace: bool = False, limit: int | None = None,
               downcast: dict | None = None) -> pd.DataFrame | None:
        return _to_pandas(self._frame).fillna(value, method, axis, inplace, limit, downcast)

    def pop(self, item: Hashable) -> Series:
        return _to_pandas(self._frame).pop(item)

    def replace(self, to_replace=None, value=lib.no_default, inplace: bool = False, limit: int | None = None,
                regex: bool = False,
                method: Literal["pad", "ffill", "bfill"] | lib.NoDefault = lib.no_default) -> pd.DataFrame | None:
        return _to_pandas(self._frame).replace(to_replace, value, inplace, limit, regex, method)

    def _replace_columnwise(self, mapping: dict[Hashable, tuple[Any, Any]], inplace: bool, regex):
        return _to_pandas(self._frame)._replace_columnwise(mapping, inplace, regex)

    def shift(self, periods: int = 1, freq: Frequency | None = None, axis: Axis = 0,
              fill_value: Hashable = lib.no_default) -> pd.DataFrame:
        return _to_pandas(self._frame).shift(periods, freq, axis, fill_value)

    def set_index(self, keys, drop: bool = True, append: bool = False, inplace: bool = False,
                  verify_integrity: bool = False) -> pd.DataFrame | None:
        return _to_pandas(self._frame).set_index(keys, drop, append, inplace, verify_integrity)

    def reset_index(self, level: IndexLabel = None, drop: bool = False, inplace: bool = False, col_level: Hashable = 0,
                    col_fill: Hashable = "", allow_duplicates: bool | lib.NoDefault = lib.no_default,
                    names: Hashable | Sequence[Hashable] = None) -> pd.DataFrame | None:
        return _to_pandas(self._frame).reset_index(level, drop, inplace, col_level, col_fill, allow_duplicates, names)

    def isna(self) -> pd.DataFrame:
        return _to_pandas(self._frame).isna()

    def isnull(self) -> pd.DataFrame:
        return _to_pandas(self._frame).isnull()

    def notna(self) -> pd.DataFrame:
        return _to_pandas(self._frame).notna()

    def notnull(self) -> pd.DataFrame:
        return _to_pandas(self._frame).notnull()

    def dropna(self, axis: Axis = 0, how: str | NoDefault = no_default, thresh: int | NoDefault = no_default,
               subset: IndexLabel = None, inplace: bool = False) -> pd.DataFrame | None:
        return _to_pandas(self._frame).dropna(axis, how, thresh, subset, inplace)

    def drop_duplicates(self, subset: Hashable | Sequence[Hashable] | None = None,
                        keep: Literal["first", "last", False] = "first", inplace: bool = False,
                        ignore_index: bool = False) -> pd.DataFrame | None:
        return _to_pandas(self._frame).drop_duplicates(subset, keep, inplace, ignore_index)

    def duplicated(self, subset: Hashable | Sequence[Hashable] | None = None,
                   keep: Literal["first", "last", False] = "first") -> Series:
        return _to_pandas(self._frame).duplicated(subset, keep)

    def sort_values(self, by: IndexLabel, axis: Axis = 0, ascending: bool | list[bool] | tuple[bool, ...] = True,
                    inplace: bool = False, kind: str = "quicksort", na_position: str = "last",
                    ignore_index: bool = False, key: ValueKeyFunc = None) -> pd.DataFrame | None:
        return _to_pandas(self._frame).sort_values(by, axis, ascending, inplace, kind, na_position, ignore_index, key)

    def sort_index(self, axis: Axis = 0, level: IndexLabel = None, ascending: bool | Sequence[bool] = True,
                   inplace: bool = False, kind: SortKind = "quicksort", na_position: NaPosition = "last",
                   sort_remaining: bool = True, ignore_index: bool = False,
                   key: IndexKeyFunc = None) -> pd.DataFrame | None:
        return _to_pandas(self._frame).sort_index(axis, level, ascending, inplace, kind, na_position, sort_remaining,
                                                  ignore_index, key)

    def value_counts(self, subset: Sequence[Hashable] | None = None, normalize: bool = False, sort: bool = True,
                     ascending: bool = False, dropna: bool = True) -> Series:
        return _to_pandas(self._frame).value_counts(subset, normalize, sort, ascending, dropna)

    def nlargest(self, n: int, columns: IndexLabel, keep: str = "first") -> pd.DataFrame:
        return _to_pandas(self._frame).nlargest(n, columns, keep)

    def nsmallest(self, n: int, columns: IndexLabel, keep: str = "first") -> pd.DataFrame:
        return _to_pandas(self._frame).nsmallest(n, columns, keep)

    def swaplevel(self, i: Axis = -2, j: Axis = -1, axis: Axis = 0) -> pd.DataFrame:
        return _to_pandas(self._frame).swaplevel(i, j, axis)

    def reorder_levels(self, order: Sequence[Axis], axis: Axis = 0) -> pd.DataFrame:
        return _to_pandas(self._frame).reorder_levels(order, axis)

    def _cmp_method(self, other, op):
        return _to_pandas(self._frame)._cmp_method(other, op)

    def _arith_method(self, other, op):
        return _to_pandas(self._frame)._arith_method(other, op)

    def _dispatch_frame_op(self, right, func: Callable, axis: int | None = None):
        return _to_pandas(self._frame)._dispatch_frame_op(right, func, axis)

    def _combine_frame(self, other: pd.DataFrame, func, fill_value=None):
        return _to_pandas(self._frame)._combine_frame(other, func, fill_value)

    def _construct_result(self, result) -> pd.DataFrame:
        return _to_pandas(self._frame)._construct_result(result)

    def __divmod__(self, other) -> tuple[pd.DataFrame, pd.DataFrame]:
        return _to_pandas(self._frame).__divmod__(other)

    def __rdivmod__(self, other) -> tuple[pd.DataFrame, pd.DataFrame]:
        return _to_pandas(self._frame).__rdivmod__(other)

    def compare(self, other: pd.DataFrame, align_axis: Axis = 1, keep_shape: bool = False, keep_equal: bool = False,
                result_names: Suffixes = ("self", "other")) -> pd.DataFrame:
        return _to_pandas(self._frame).compare(other, align_axis, keep_shape, keep_equal, result_names)

    def combine(self, other: pd.DataFrame, func: Callable[[Series, Series], Series | Hashable], fill_value=None,
                overwrite: bool = True) -> pd.DataFrame:
        return _to_pandas(self._frame).combine(other, func, fill_value, overwrite)

    def combine_first(self, other: pd.DataFrame) -> pd.DataFrame:
        return _to_pandas(self._frame).combine_first(other)

    def update(self, other, join: str = "left", overwrite: bool = True, filter_func=None,
               errors: str = "ignore") -> None:
        _to_pandas(self._frame).update(other, join, overwrite, filter_func, errors)

    def groupby(self, by=None, axis: Axis = 0, level: IndexLabel | None = None, as_index: bool = True,
                sort: bool = True, group_keys: bool | lib.NoDefault = no_default,
                squeeze: bool | lib.NoDefault = no_default, observed: bool = False,
                dropna: bool = True) -> pd.DataFrameGroupBy:
        return _to_pandas(self._frame).groupby(by, axis, level, as_index, sort, group_keys, squeeze, observed, dropna)

    def pivot(self, index=None, columns=None, values=None) -> pd.DataFrame:
        return _to_pandas(self._frame).pivot(index, columns, values)

    def pivot_table(self, values=None, index=None, columns=None, aggfunc="mean", fill_value=None, margins=False,
                    dropna=True, margins_name="All", observed=False, sort=True) -> pd.DataFrame:
        return _to_pandas(self._frame).pivot_table(values, index, columns, aggfunc, fill_value, margins, dropna,
                                                   margins_name, observed,
                                                   sort)

    def stack(self, level: Level = -1, dropna: bool = True):
        return _to_pandas(self._frame).stack(level, dropna)

    def explode(self, column: IndexLabel, ignore_index: bool = False) -> pd.DataFrame:
        return _to_pandas(self._frame).explode(column, ignore_index)

    def unstack(self, level: Level = -1, fill_value=None):
        return _to_pandas(self._frame).unstack(level, fill_value)

    def melt(self, id_vars=None, value_vars=None, var_name=None, value_name="value", col_level: Level = None,
             ignore_index: bool = True) -> pd.DataFrame:
        return _to_pandas(self._frame).melt(id_vars, value_vars, var_name, value_name, col_level, ignore_index)

    def diff(self, periods: int = 1, axis: Axis = 0) -> pd.DataFrame:
        return _to_pandas(self._frame).diff(periods, axis)

    def _gotitem(self, key: IndexLabel, ndim: int,
                 subset: pd.DataFrame | Series | None = None) -> pd.DataFrame | Series:
        return _to_pandas(self._frame)._gotitem(key, ndim, subset)

    def aggregate(self, func=None, axis: Axis = 0, *args, **kwargs):
        return _to_pandas(self._frame).aggregate(func, axis, *args, **kwargs)

    def any(self, axis: Axis = 0, bool_only: bool | None = None, skipna: bool = True, level: Level = None,
            **kwargs) -> pd.DataFrame | Series:
        return _to_pandas(self._frame).any(axis, bool_only, skipna, level, **kwargs)

    def transform(self, func: AggFuncType, axis: Axis = 0, *args, **kwargs) -> pd.DataFrame:
        return _to_pandas(self._frame).transform(func, axis, *args, **kwargs)

    def apply(self, func: AggFuncType, axis: Axis = 0, raw: bool = False,
              result_type: Literal["expand", "reduce", "broadcast"] | None = None, args=(), **kwargs):
        return _to_pandas(self._frame).apply(func, axis, raw, result_type, args, **kwargs)

    def applymap(self, func: PythonFuncType, na_action: str | None = None, **kwargs) -> pd.DataFrame:
        return _to_pandas(self._frame).applymap(func, na_action, **kwargs)

    def append(self, other, ignore_index: bool = False, verify_integrity: bool = False,
               sort: bool = False) -> pd.DataFrame:
        return _to_pandas(self._frame).append(other, ignore_index, verify_integrity, sort)

    def _append(self, other, ignore_index: bool = False, verify_integrity: bool = False,
                sort: bool = False) -> pd.DataFrame:
        return _to_pandas(self._frame)._append(other, ignore_index, verify_integrity, sort)

    def join(self, other: pd.DataFrame | Series | list[pd.DataFrame | Series], on: IndexLabel | None = None,
             how: str = "left", lsuffix: str = "", rsuffix: str = "", sort: bool = False,
             validate: str | None = None) -> pd.DataFrame:
        return _to_pandas(self._frame).join(other, on, how, lsuffix, rsuffix, sort, validate)

    def _join_compat(self, other: pd.DataFrame | Series | Iterable[pd.DataFrame | Series], on: IndexLabel | None = None,
                     how: str = "left", lsuffix: str = "", rsuffix: str = "", sort: bool = False,
                     validate: str | None = None):
        return _to_pandas(self._frame)._join_compat(other, on, how, lsuffix, rsuffix, sort, validate)

    def merge(self, right: pd.DataFrame | Series, how: str = "inner", on: IndexLabel | None = None,
              left_on: IndexLabel | None = None, right_on: IndexLabel | None = None, left_index: bool = False,
              right_index: bool = False, sort: bool = False, suffixes: Suffixes = ("_x", "_y"), copy: bool = True,
              indicator: bool = False, validate: str | None = None) -> pd.DataFrame:
        return _to_pandas(self._frame).merge(right, how, on, left_on, right_on, left_index, right_index, sort, suffixes,
                                             copy,
                                             indicator, validate)

    def round(self, decimals: int | dict[IndexLabel, int] | Series = 0, *args, **kwargs) -> pd.DataFrame:
        return _to_pandas(self._frame).round(decimals, *args, **kwargs)

    def corr(self, method: str | Callable[[np.ndarray, np.ndarray], float] = "pearson", min_periods: int = 1,
             numeric_only: bool | lib.NoDefault = lib.no_default) -> pd.DataFrame:
        return _to_pandas(self._frame).corr(method, min_periods, numeric_only)

    def cov(self, min_periods: int | None = None, ddof: int | None = 1,
            numeric_only: bool | lib.NoDefault = lib.no_default) -> pd.DataFrame:
        return _to_pandas(self._frame).cov(min_periods, ddof, numeric_only)

    def corrwith(self, other: pd.DataFrame | Series, axis: Axis = 0, drop: bool = False,
                 method: Literal["pearson", "kendall", "spearman"]
                         | Callable[[np.ndarray, np.ndarray], float] = "pearson",
                 numeric_only: bool | lib.NoDefault = lib.no_default) -> Series:
        return _to_pandas(self._frame).corrwith(other, axis, drop, method, numeric_only)

    def count(self, axis: Axis = 0, level: Level = None, numeric_only: bool = False):
        return _to_pandas(self._frame).count(axis, level, numeric_only)

    def _count_level(self, level: Level, axis: int = 0, numeric_only: bool = False):
        return _to_pandas(self._frame)._count_level(level, axis, numeric_only)

    def _reduce(self, op, name: str, *, axis: Axis = 0, skipna: bool = True, numeric_only: bool | None = None,
                filter_type=None, **kwds):
        return _to_pandas(self._frame)._reduce(op, name, axis=axis, skipna=skipna, numeric_only=numeric_only,
                                               filter_type=filter_type,
                                               **kwds)

    def _reduce_axis1(self, name: str, func, skipna: bool) -> Series:
        return _to_pandas(self._frame)._reduce_axis1(name, func, skipna)

    def nunique(self, axis: Axis = 0, dropna: bool = True) -> Series:
        return _to_pandas(self._frame).nunique(axis, dropna)

    def idxmin(self, axis: Axis = 0, skipna: bool = True, numeric_only: bool = False) -> Series:
        return _to_pandas(self._frame).idxmin(axis, skipna, numeric_only)

    def idxmax(self, axis: Axis = 0, skipna: bool = True, numeric_only: bool = False) -> Series:
        return _to_pandas(self._frame).idxmax(axis, skipna, numeric_only)

    def _get_agg_axis(self, axis_num: int) -> Index:
        return _to_pandas(self._frame)._get_agg_axis(axis_num)

    def mode(self, axis: Axis = 0, numeric_only: bool = False, dropna: bool = True) -> pd.DataFrame:
        return _to_pandas(self._frame).mode(axis, numeric_only, dropna)

    def quantile(self, q: float | AnyArrayLike | Sequence[float] = 0.5, axis: Axis = 0,
                 numeric_only: bool | lib.NoDefault = no_default, interpolation: QuantileInterpolation = "linear",
                 method: Literal["single", "table"] = "single") -> Series | pd.DataFrame:
        return _to_pandas(self._frame).quantile(q, axis, numeric_only, interpolation, method)

    def asfreq(self, freq: Frequency, method: FillnaOptions | None = None, how: str | None = None,
               normalize: bool = False, fill_value: Hashable = None) -> pd.DataFrame:
        return _to_pandas(self._frame).asfreq(freq, method, how, normalize, fill_value)

    def resample(self, rule, axis: Axis = 0, closed: str | None = None, label: str | None = None,
                 convention: str = "start", kind: str | None = None, loffset=None, base: int | None = None,
                 on: Level = None, level: Level = None, origin: str | TimestampConvertibleTypes = "start_day",
                 offset: TimedeltaConvertibleTypes | None = None,
                 group_keys: bool | lib.NoDefault = no_default) -> Resampler:
        return _to_pandas(self._frame).resample(rule, axis, closed, label, convention, kind, loffset, base, on, level,
                                                origin, offset,
                                                group_keys)

    def to_timestamp(self, freq: Frequency | None = None, how: str = "start", axis: Axis = 0,
                     copy: bool = True) -> pd.DataFrame:
        return _to_pandas(self._frame).to_timestamp(freq, how, axis, copy)

    def to_period(self, freq: Frequency | None = None, axis: Axis = 0, copy: bool = True) -> pd.DataFrame:
        return _to_pandas(self._frame).to_period(freq, axis, copy)

    def isin(self, values: Series | pd.DataFrame | Sequence | Mapping) -> pd.DataFrame:
        return _to_pandas(self._frame).isin(values)

    @property
    def _AXIS_NUMBERS(self) -> dict[str, int]:
        return _to_pandas(self._frame)._AXIS_NUMBERS

    @property
    def _AXIS_NAMES(self) -> dict[int, str]:
        return _to_pandas(self._frame)._AXIS_NAMES

    def _to_dict_of_blocks(self, copy: bool = True):
        return _to_pandas(self._frame)._to_dict_of_blocks(copy)

    @property
    def values(self) -> np.ndarray:
        return _to_pandas(self._frame).values

    def ffill(self, axis: None | Axis = None, inplace: bool = False, limit: None | int = None,
              downcast: dict | None = None) -> pd.DataFrame | None:
        return _to_pandas(self._frame).ffill(axis, inplace, limit, downcast)

    def bfill(self, axis: None | Axis = None, inplace: bool = False, limit: None | int = None,
              downcast=None) -> pd.DataFrame | None:
        return _to_pandas(self._frame).bfill(axis, inplace, limit, downcast)

    def clip(self: pd.DataFrame, lower: float | None = None, upper: float | None = None, axis: Axis | None = None,
             inplace: bool = False, *args, **kwargs) -> pd.DataFrame | None:
        return _to_pandas(self._frame).clip(lower, upper, axis, inplace, *args, **kwargs)

    def interpolate(self: pd.DataFrame, method: str = "linear", axis: Axis = 0, limit: int | None = None,
                    inplace: bool = False, limit_direction: str | None = None, limit_area: str | None = None,
                    downcast: str | None = None, **kwargs) -> pd.DataFrame | None:
        return _to_pandas(self._frame).interpolate(method, axis, limit, inplace, limit_direction, limit_area, downcast,
                                                   **kwargs)

    def where(self, cond, other=lib.no_default, inplace: bool = False, axis: Axis | None = None, level: Level = None,
              errors: IgnoreRaise | lib.NoDefault = "raise",
              try_cast: bool | lib.NoDefault = lib.no_default) -> pd.DataFrame | None:
        return _to_pandas(self._frame).where(cond, other, inplace, axis, level, errors, try_cast)

    def mask(self, cond, other=np.nan, inplace: bool = False, axis: Axis | None = None, level: Level = None,
             errors: IgnoreRaise | lib.NoDefault = "raise",
             try_cast: bool | lib.NoDefault = lib.no_default) -> pd.DataFrame | None:
        return _to_pandas(self._frame).mask(cond, other, inplace, axis, level, errors, try_cast)

    @classmethod
    def _init_mgr(cls, mgr: Manager, axes, dtype: Dtype | None = None, copy: bool_t = False) -> Manager:
        return _to_pandas(self._frame)._init_mgr(mgr, axes, dtype, copy)

    def _as_manager(self: NDFrameT, typ: str, copy: bool_t = True) -> NDFrameT:
        return _to_pandas(self._frame)._as_manager(typ, copy)

    @property
    def attrs(self) -> dict[Hashable, Any]:
        return _to_pandas(self._frame).attrs

    @property
    def flags(self) -> Flags:
        return _to_pandas(self._frame).flags

    def set_flags(self: NDFrameT, *, copy: bool_t = False, allows_duplicate_labels: bool_t | None = None) -> NDFrameT:
        return _to_pandas(self._frame).set_flags(copy=copy, allows_duplicate_labels=allows_duplicate_labels)

    @classmethod
    def _validate_dtype(cls, dtype) -> DtypeObj | None:
        return _to_pandas(self._frame)._validate_dtype(dtype)

    @property
    def _data(self):
        return _to_pandas(self._frame)._data

    def _construct_axes_dict(self, axes=None, **kwargs):
        return _to_pandas(self._frame)._construct_axes_dict(axes, **kwargs)

    @classmethod
    def _construct_axes_from_arguments(cls, args, kwargs, require_all: bool_t = False, sentinel=None):
        return _to_pandas(self._frame)._construct_axes_from_arguments(args, kwargs, require_all, sentinel)

    @classmethod
    def _get_axis_number(cls, axis: Axis) -> int:
        return _to_pandas(self._frame)._get_axis_number(axis)

    @classmethod
    def _get_axis_name(cls, axis: Axis) -> str:
        return _to_pandas(self._frame)._get_axis_name(axis)

    def _get_axis(self, axis: Axis) -> Index:
        return _to_pandas(self._frame)._get_axis(axis)

    @classmethod
    def _get_block_manager_axis(cls, axis: Axis) -> int:
        return _to_pandas(self._frame)._get_block_manager_axis(axis)

    def _get_axis_resolvers(self, axis: str) -> dict[str, Series | MultiIndex]:
        return _to_pandas(self._frame)._get_axis_resolvers(axis)

    def _get_index_resolvers(self) -> dict[Hashable, Series | MultiIndex]:
        return _to_pandas(self._frame)._get_index_resolvers()

    def _get_cleaned_column_resolvers(self) -> dict[Hashable, Series]:
        return _to_pandas(self._frame)._get_cleaned_column_resolvers()

    @property
    def _info_axis(self) -> Index:
        return _to_pandas(self._frame)._info_axis

    @property
    def _stat_axis(self) -> Index:
        return _to_pandas(self._frame)._stat_axis

    @property
    def ndim(self) -> int:
        return _to_pandas(self._frame).ndim

    @property
    def size(self) -> int:
        return _to_pandas(self._frame).size

    def _set_axis_nocheck(self, labels, axis: Axis, inplace: bool_t, copy: bool_t):
        return _to_pandas(self._frame)._set_axis_nocheck(labels, axis, inplace, copy)

    def _set_axis(self, axis: int, labels: AnyArrayLike | list) -> None:
        _to_pandas(self._frame)._set_axis(axis, labels)

    def swapaxes(self: NDFrameT, axis1: Axis, axis2: Axis, copy: bool_t = True) -> NDFrameT:
        return _to_pandas(self._frame).swapaxes(axis1, axis2, copy)

    def droplevel(self: NDFrameT, level: IndexLabel, axis: Axis = 0) -> NDFrameT:
        return _to_pandas(self._frame).droplevel(level, axis)

    def squeeze(self, axis=None):
        return _to_pandas(self._frame).squeeze(axis)

    def _rename(self: NDFrameT, mapper: Renamer | None = None, *, index: Renamer | None = None,
                columns: Renamer | None = None, axis: Axis | None = None, copy: bool_t | None = None,
                inplace: bool_t = False, level: Level | None = None, errors: str = "ignore") -> NDFrameT | None:
        return _to_pandas(self._frame)._rename(mapper, index=index, columns=columns, axis=axis, copy=copy,
                                               inplace=inplace, level=level,
                                               errors=errors)

    def rename_axis(self: NDFrameT, mapper: IndexLabel | lib.NoDefault = lib.no_default, inplace: bool_t = False,
                    **kwargs) -> NDFrameT | None:
        return _to_pandas(self._frame).rename_axis(mapper, inplace, **kwargs)

    def _set_axis_name(self, name, axis=0, inplace=False):
        return _to_pandas(self._frame)._set_axis_name(name, axis, inplace)

    def _indexed_same(self, other) -> bool_t:
        return _to_pandas(self._frame)._indexed_same(other)

    def equals(self, other: object) -> bool_t:
        return _to_pandas(self._frame).equals(other)

    def __neg__(self: NDFrameT) -> NDFrameT:
        return _to_pandas(self._frame).__neg__()

    def __pos__(self: NDFrameT) -> NDFrameT:
        return _to_pandas(self._frame).__pos__()

    def __invert__(self: NDFrameT) -> NDFrameT:
        return _to_pandas(self._frame).__invert__()

    def __nonzero__(self) -> NoReturn:
        return _to_pandas(self._frame).__nonzero__()

    def bool(self) -> bool_t:
        return _to_pandas(self._frame).bool()

    def abs(self: NDFrameT) -> NDFrameT:
        return _to_pandas(self._frame).abs()

    def __abs__(self: NDFrameT) -> NDFrameT:
        return _to_pandas(self._frame).__abs__()

    def __round__(self: NDFrameT, decimals: int = 0) -> NDFrameT:
        return _to_pandas(self._frame).__round__(decimals)

    def _is_level_reference(self, key: Level, axis=0) -> bool_t:
        return _to_pandas(self._frame)._is_level_reference(key, axis)

    def _is_label_reference(self, key: Level, axis=0) -> bool_t:
        return _to_pandas(self._frame)._is_label_reference(key, axis)

    def _is_label_or_level_reference(self, key: Level, axis: int = 0) -> bool_t:
        return _to_pandas(self._frame)._is_label_or_level_reference(key, axis)

    def _check_label_or_level_ambiguity(self, key: Level, axis: int = 0) -> None:
        _to_pandas(self._frame)._check_label_or_level_ambiguity(key, axis)

    def _get_label_or_level_values(self, key: Level, axis: int = 0) -> ArrayLike:
        return _to_pandas(self._frame)._get_label_or_level_values(key, axis)

    def _drop_labels_or_levels(self, keys, axis: int = 0):
        return _to_pandas(self._frame)._drop_labels_or_levels(keys, axis)

    def __iter__(self):
        return _to_pandas(self._frame).__iter__()

    def keys(self) -> Index:
        return _to_pandas(self._frame).keys()

    def __contains__(self, key) -> bool_t:
        return _to_pandas(self._frame).__contains__(key)

    @property
    def empty(self) -> bool_t:
        return _to_pandas(self._frame).empty

    def __array__(self, dtype: npt.DTypeLike | None = None) -> np.ndarray:
        return _to_pandas(self._frame).__array__(dtype)

    def __array_wrap__(self, result: np.ndarray, context: tuple[Callable, tuple[Any, ...], int] | None = None):
        return _to_pandas(self._frame).__array_wrap__(result, context)

    def __array_ufunc__(self, ufunc: np.ufunc, method: str, *inputs: Any, **kwargs: Any):
        return _to_pandas(self._frame).__array_ufunc__(ufunc, method, *inputs, **kwargs)

    def __getstate__(self) -> dict[str, Any]:
        return _to_pandas(self._frame).__getstate__()

    def __setstate__(self, state) -> None:
        _to_pandas(self._frame).__setstate__(state)

    def _repr_latex_(self):
        return _to_pandas(self._frame)._repr_latex_()

    def _repr_data_resource_(self):
        return _to_pandas(self._frame)._repr_data_resource_()

    def to_excel(self, excel_writer, sheet_name: str = "Sheet1", na_rep: str = "", float_format: str | None = None,
                 columns: Sequence[Hashable] | None = None, header: Sequence[Hashable] | bool_t = True,
                 index: bool_t = True, index_label: IndexLabel = None, startrow: int = 0, startcol: int = 0,
                 engine: str | None = None, merge_cells: bool_t = True, encoding: lib.NoDefault = lib.no_default,
                 inf_rep: str = "inf", verbose: lib.NoDefault = lib.no_default,
                 freeze_panes: tuple[int, int] | None = None, storage_options: StorageOptions = None) -> None:
        _to_pandas(self._frame).to_excel(excel_writer, sheet_name, na_rep, float_format, columns, header, index,
                                         index_label, startrow,
                                         startcol, engine, merge_cells, encoding, inf_rep, verbose, freeze_panes,
                                         storage_options)

    def to_json(self, path_or_buf: FilePath | WriteBuffer[bytes] | WriteBuffer[str] | None = None,
                orient: str | None = None, date_format: str | None = None, double_precision: int = 10,
                force_ascii: bool_t = True, date_unit: str = "ms",
                default_handler: Callable[[Any], JSONSerializable] | None = None, lines: bool_t = False,
                compression: CompressionOptions = "infer", index: bool_t = True, indent: int | None = None,
                storage_options: StorageOptions = None) -> str | None:
        return _to_pandas(self._frame).to_json(path_or_buf, orient, date_format, double_precision, force_ascii,
                                               date_unit,
                                               default_handler, lines, compression, index, indent, storage_options)

    def to_hdf(self, path_or_buf: FilePath | HDFStore, key: str, mode: str = "a", complevel: int | None = None,
               complib: str | None = None, append: bool_t = False, format: str | None = None, index: bool_t = True,
               min_itemsize: int | dict[str, int] | None = None, nan_rep=None, dropna: bool_t | None = None,
               data_columns: Literal[True] | list[str] | None = None, errors: str = "strict",
               encoding: str = "UTF-8") -> None:
        _to_pandas(self._frame).to_hdf(path_or_buf, key, mode, complevel, complib, append, format, index, min_itemsize,
                                       nan_rep, dropna,
                                       data_columns, errors, encoding)

    def to_sql(self, name: str, con, schema: str | None = None, if_exists: str = "fail", index: bool_t = True,
               index_label: IndexLabel = None, chunksize: int | None = None, dtype: DtypeArg | None = None,
               method: str | None = None) -> int | None:
        return _to_pandas(self._frame).to_sql(name, con, schema, if_exists, index, index_label, chunksize, dtype,
                                              method)

    def to_pickle(self, path: FilePath | WriteBuffer[bytes], compression: CompressionOptions = "infer",
                  protocol: int = pickle.HIGHEST_PROTOCOL, storage_options: StorageOptions = None) -> None:
        _to_pandas(self._frame).to_pickle(path, compression, protocol, storage_options)

    def to_clipboard(self, excel: bool_t = True, sep: str | None = None, **kwargs) -> None:
        _to_pandas(self._frame).to_clipboard(excel, sep, **kwargs)

    def to_xarray(self):
        return _to_pandas(self._frame).to_xarray()

    def to_latex(self, buf: FilePath | WriteBuffer[str] | None = None, columns: Sequence[Hashable] | None = None,
                 col_space: ColspaceArgType | None = None, header: bool_t | Sequence[str] = True, index: bool_t = True,
                 na_rep: str = "NaN", formatters: FormattersType | None = None,
                 float_format: FloatFormatType | None = None, sparsify: bool_t | None = None,
                 index_names: bool_t = True, bold_rows: bool_t = False, column_format: str | None = None,
                 longtable: bool_t | None = None, escape: bool_t | None = None, encoding: str | None = None,
                 decimal: str = ".", multicolumn: bool_t | None = None, multicolumn_format: str | None = None,
                 multirow: bool_t | None = None, caption: str | tuple[str, str] | None = None, label: str | None = None,
                 position: str | None = None) -> str | None:
        return _to_pandas(self._frame).to_latex(buf, columns, col_space, header, index, na_rep, formatters,
                                                float_format, sparsify,
                                                index_names, bold_rows, column_format, longtable, escape, encoding,
                                                decimal,
                                                multicolumn, multicolumn_format, multirow, caption, label, position)

    def to_csv(self, path_or_buf: FilePath | WriteBuffer[bytes] | WriteBuffer[str] | None = None, sep: str = ",",
               na_rep: str = "", float_format: str | Callable | None = None, columns: Sequence[Hashable] | None = None,
               header: bool_t | list[str] = True, index: bool_t = True, index_label: IndexLabel | None = None,
               mode: str = "w", encoding: str | None = None, compression: CompressionOptions = "infer",
               quoting: int | None = None, quotechar: str = '"', lineterminator: str | None = None,
               chunksize: int | None = None, date_format: str | None = None, doublequote: bool_t = True,
               escapechar: str | None = None, decimal: str = ".", errors: str = "strict",
               storage_options: StorageOptions = None) -> str | None:
        return _to_pandas(self._frame).to_csv(path_or_buf, sep, na_rep, float_format, columns, header, index,
                                              index_label, mode,
                                              encoding, compression, quoting, quotechar, lineterminator, chunksize,
                                              date_format,
                                              doublequote, escapechar, decimal, errors, storage_options)

    def _maybe_update_cacher(self, clear: bool_t = False, verify_is_copy: bool_t = True,
                             inplace: bool_t = False) -> None:
        _to_pandas(self._frame)._maybe_update_cacher(clear, verify_is_copy, inplace)

    def take(self: NDFrameT, indices, axis=0, is_copy: bool_t | None = None, **kwargs) -> NDFrameT:
        return _to_pandas(self._frame).take(indices, axis, is_copy, **kwargs)

    def _take(self: NDFrameT, indices, axis=0, convert_indices: bool_t = True) -> NDFrameT:
        return _to_pandas(self._frame)._take(indices, axis, convert_indices)

    def _take_with_is_copy(self: NDFrameT, indices, axis=0) -> NDFrameT:
        return _to_pandas(self._frame)._take_with_is_copy(indices, axis)

    def xs(self: NDFrameT, key: IndexLabel, axis: Axis = 0, level: IndexLabel = None,
           drop_level: bool_t = True) -> NDFrameT:
        return _to_pandas(self._frame).xs(key, axis, level, drop_level)

    def _slice(self: NDFrameT, slobj: slice, axis=0) -> NDFrameT:
        return _to_pandas(self._frame)._slice(slobj, axis)

    def _set_is_copy(self, ref: NDFrame, copy: bool_t = True) -> None:
        _to_pandas(self._frame)._set_is_copy(ref, copy)

    def _check_is_chained_assignment_possible(self) -> bool_t:
        return _to_pandas(self._frame)._check_is_chained_assignment_possible()

    def _check_setitem_copy(self, t="setting", force=False):
        _to_pandas(self._frame)._check_setitem_copy(t, force)

    def __delitem__(self, key) -> None:
        _to_pandas(self._frame).__delitem__(key)

    def _check_inplace_and_allows_duplicate_labels(self, inplace):
        return _to_pandas(self._frame)._check_inplace_and_allows_duplicate_labels(inplace)

    def get(self, key, default=None):
        return _to_pandas(self._frame).get(key, default)

    @property
    def _is_view(self) -> bool_t:
        return _to_pandas(self._frame)._is_view

    def reindex_like(self: NDFrameT, other, method: str | None = None, copy: bool_t = True, limit=None,
                     tolerance=None) -> NDFrameT:
        return _to_pandas(self._frame).reindex_like(other, method, copy, limit, tolerance)

    def _drop_axis(self: NDFrameT, labels, axis, level=None, errors: IgnoreRaise = "raise",
                   only_slice: bool_t = False) -> NDFrameT:
        return _to_pandas(self._frame)._drop_axis(labels, axis, level, errors, only_slice)

    def _update_inplace(self, result, verify_is_copy: bool_t = True) -> None:
        _to_pandas(self._frame)._update_inplace(result, verify_is_copy)

    def add_prefix(self: NDFrameT, prefix: str) -> NDFrameT:
        return _to_pandas(self._frame).add_prefix(prefix)

    def add_suffix(self: NDFrameT, suffix: str) -> NDFrameT:
        return _to_pandas(self._frame).add_suffix(suffix)

    def _needs_reindex_multi(self, axes, method, level) -> bool_t:
        return _to_pandas(self._frame)._needs_reindex_multi(axes, method, level)

    def _reindex_with_indexers(self: NDFrameT, reindexers, fill_value=None, copy: bool_t = False,
                               allow_dups: bool_t = False) -> NDFrameT:
        return _to_pandas(self._frame)._reindex_with_indexers(reindexers, fill_value, copy, allow_dups)

    def filter(self: NDFrameT, items=None, like: str | None = None, regex: str | None = None, axis=None) -> NDFrameT:
        return _to_pandas(self._frame).filter(items, like, regex, axis)

    def head(self: NDFrameT, n: int = 5) -> NDFrameT:
        return _to_pandas(self._frame).head(n)

    def tail(self: NDFrameT, n: int = 5) -> NDFrameT:
        return _to_pandas(self._frame).tail(n)

    def sample(self: NDFrameT, n: int | None = None, frac: float | None = None, replace: bool_t = False, weights=None,
               random_state: RandomState | None = None, axis: Axis | None = None,
               ignore_index: bool_t = False) -> NDFrameT:
        return _to_pandas(self._frame).sample(n, frac, replace, weights, random_state, axis, ignore_index)

    def pipe(self, func: Callable[..., T] | tuple[Callable[..., T], str], *args, **kwargs) -> T:
        return _to_pandas(self._frame).pipe(func, *args, **kwargs)

    def __finalize__(self: NDFrameT, other, method: str | None = None, **kwargs) -> NDFrameT:
        return _to_pandas(self._frame).__finalize__(other, method, **kwargs)

    def __getattr__(self, name: str):
        return _to_pandas(self._frame).__getattr__(name)

    def __setattr__(self, name: str, value) -> None:
        _to_pandas(self._frame).__setattr__(name, value)

    def _dir_additions(self) -> set[str]:
        return _to_pandas(self._frame)._dir_additions()

    def _protect_consolidate(self, f):
        return _to_pandas(self._frame)._protect_consolidate(f)

    def _consolidate_inplace(self) -> None:
        _to_pandas(self._frame)._consolidate_inplace()

    def _consolidate(self):
        return _to_pandas(self._frame)._consolidate()

    @property
    def _is_mixed_type(self) -> bool_t:
        return _to_pandas(self._frame)._is_mixed_type

    def _check_inplace_setting(self, value) -> bool_t:
        return _to_pandas(self._frame)._check_inplace_setting(value)

    def _get_numeric_data(self: NDFrameT) -> NDFrameT:
        return _to_pandas(self._frame)._get_numeric_data()

    def _get_bool_data(self):
        return _to_pandas(self._frame)._get_bool_data()

    @property
    def dtypes(self):
        return _to_pandas(self._frame).dtypes

    def astype(self: NDFrameT, dtype, copy: bool_t = True, errors: IgnoreRaise = "raise") -> NDFrameT:
        return _to_pandas(self._frame).astype(dtype, copy, errors)

    def copy(self: NDFrameT, deep: bool_t | None = True) -> NDFrameT:
        return _to_pandas(self._frame).copy(deep)

    def __copy__(self: NDFrameT, deep: bool_t = True) -> NDFrameT:
        return _to_pandas(self._frame).__copy__(deep)

    def __deepcopy__(self: NDFrameT, memo=None) -> NDFrameT:
        return _to_pandas(self._frame).__deepcopy__(memo)

    def _convert(self: NDFrameT, datetime: bool_t = False, numeric: bool_t = False,
                 timedelta: bool_t = False) -> NDFrameT:
        return _to_pandas(self._frame)._convert(datetime, numeric, timedelta)

    def infer_objects(self: NDFrameT) -> NDFrameT:
        return _to_pandas(self._frame).infer_objects()

    def convert_dtypes(self: NDFrameT, infer_objects: bool_t = True, convert_string: bool_t = True,
                       convert_integer: bool_t = True, convert_boolean: bool_t = True,
                       convert_floating: bool_t = True) -> NDFrameT:
        return _to_pandas(self._frame).convert_dtypes(infer_objects, convert_string, convert_integer, convert_boolean,
                                                      convert_floating)

    def asof(self, where, subset=None):
        return _to_pandas(self._frame).asof(where, subset)

    def _clip_with_scalar(self, lower, upper, inplace: bool_t = False):
        return _to_pandas(self._frame)._clip_with_scalar(lower, upper, inplace)

    def _clip_with_one_bound(self, threshold, method, axis, inplace):
        return _to_pandas(self._frame)._clip_with_one_bound(threshold, method, axis, inplace)

    def at_time(self: NDFrameT, time, asof: bool_t = False, axis=None) -> NDFrameT:
        return _to_pandas(self._frame).at_time(time, asof, axis)

    def between_time(self: NDFrameT, start_time, end_time, include_start: bool_t | lib.NoDefault = lib.no_default,
                     include_end: bool_t | lib.NoDefault = lib.no_default, inclusive: IntervalClosedType | None = None,
                     axis=None) -> NDFrameT:
        return _to_pandas(self._frame).between_time(start_time, end_time, include_start, include_end, inclusive, axis)

    def first(self: NDFrameT, offset) -> NDFrameT:
        return _to_pandas(self._frame).first(offset)

    def last(self: NDFrameT, offset) -> NDFrameT:
        return _to_pandas(self._frame).last(offset)

    def rank(self: NDFrameT, axis=0, method: str = "average",
             numeric_only: bool_t | None | lib.NoDefault = lib.no_default, na_option: str = "keep",
             ascending: bool_t = True, pct: bool_t = False) -> NDFrameT:
        return _to_pandas(self._frame).rank(axis, method, numeric_only, na_option, ascending, pct)

    def _align_frame(self, other, join="outer", axis=None, level=None, copy: bool_t = True, fill_value=None,
                     method=None, limit=None, fill_axis=0):
        return _to_pandas(self._frame)._align_frame(other, join, axis, level, copy, fill_value, method, limit,
                                                    fill_axis)

    def _align_series(self, other, join="outer", axis=None, level=None, copy: bool_t = True, fill_value=None,
                      method=None, limit=None, fill_axis=0):
        return _to_pandas(self._frame)._align_series(other, join, axis, level, copy, fill_value, method, limit,
                                                     fill_axis)

    def _where(self, cond, other=lib.no_default, inplace=False, axis=None, level=None):
        return _to_pandas(self._frame)._where(cond, other, inplace, axis, level)

    def slice_shift(self: NDFrameT, periods: int = 1, axis=0) -> NDFrameT:
        return _to_pandas(self._frame).slice_shift(periods, axis)

    def tshift(self: NDFrameT, periods: int = 1, freq=None, axis: Axis = 0) -> NDFrameT:
        return _to_pandas(self._frame).tshift(periods, freq, axis)

    def truncate(self: NDFrameT, before=None, after=None, axis=None, copy: bool_t = True) -> NDFrameT:
        return _to_pandas(self._frame).truncate(before, after, axis, copy)

    def tz_convert(self: NDFrameT, tz, axis=0, level=None, copy: bool_t = True) -> NDFrameT:
        return _to_pandas(self._frame).tz_convert(tz, axis, level, copy)

    def tz_localize(self: NDFrameT, tz, axis=0, level=None, copy: bool_t = True, ambiguous="raise",
                    nonexistent: str = "raise") -> NDFrameT:
        return _to_pandas(self._frame).tz_localize(tz, axis, level, copy, ambiguous, nonexistent)

    def describe(self: NDFrameT, percentiles=None, include=None, exclude=None,
                 datetime_is_numeric: bool_t = False) -> NDFrameT:
        return _to_pandas(self._frame).describe(percentiles, include, exclude, datetime_is_numeric)

    def pct_change(self: NDFrameT, periods=1, fill_method="pad", limit=None, freq=None, **kwargs) -> NDFrameT:
        return _to_pandas(self._frame).pct_change(periods, fill_method, limit, freq, **kwargs)

    def _agg_by_level(self, name: str, axis: Axis = 0, level: Level = 0, skipna: bool_t = True, **kwargs):
        return _to_pandas(self._frame)._agg_by_level(name, axis, level, skipna, **kwargs)

    def _logical_func(self, name: str, func, axis: Axis = 0, bool_only: bool_t | None = None, skipna: bool_t = True,
                      level: Level | None = None, **kwargs) -> Series | bool_t:
        return _to_pandas(self._frame)._logical_func(name, func, axis, bool_only, skipna, level, **kwargs)

    def all(self, axis: Axis = 0, bool_only: bool_t | None = None, skipna: bool_t = True, level: Level | None = None,
            **kwargs) -> Series | bool_t:
        return _to_pandas(self._frame).all(axis, bool_only, skipna, level, **kwargs)

    def _accum_func(self, name: str, func, axis: Axis | None = None, skipna: bool_t = True, *args, **kwargs):
        return _to_pandas(self._frame)._accum_func(name, func, axis, skipna, *args, **kwargs)

    def cummax(self, axis: Axis | None = None, skipna: bool_t = True, *args, **kwargs):
        return _to_pandas(self._frame).cummax(axis, skipna, *args, **kwargs)

    def cummin(self, axis: Axis | None = None, skipna: bool_t = True, *args, **kwargs):
        return _to_pandas(self._frame).cummin(axis, skipna, *args, **kwargs)

    def cumsum(self, axis: Axis | None = None, skipna: bool_t = True, *args, **kwargs):
        return _to_pandas(self._frame).cumsum(axis, skipna, *args, **kwargs)

    def cumprod(self, axis: Axis | None = None, skipna: bool_t = True, *args, **kwargs):
        return _to_pandas(self._frame).cumprod(axis, skipna, *args, **kwargs)

    def _stat_function_ddof(self, name: str, func, axis: Axis | None = None, skipna: bool_t = True,
                            level: Level | None = None, ddof: int = 1, numeric_only: bool_t | None = None,
                            **kwargs) -> Series | float:
        return _to_pandas(self._frame)._stat_function_ddof(name, func, axis, skipna, level, ddof, numeric_only,
                                                           **kwargs)

    def sem(self, axis: Axis | None = None, skipna: bool_t = True, level: Level | None = None, ddof: int = 1,
            numeric_only: bool_t | None = None, **kwargs) -> Series | float:
        return _to_pandas(self._frame).sem(axis, skipna, level, ddof, numeric_only, **kwargs)

    def var(self, axis: Axis | None = None, skipna: bool_t = True, level: Level | None = None, ddof: int = 1,
            numeric_only: bool_t | None = None, **kwargs) -> Series | float:
        return _to_pandas(self._frame).var(axis, skipna, level, ddof, numeric_only, **kwargs)

    def std(self, axis: Axis | None = None, skipna: bool_t = True, level: Level | None = None, ddof: int = 1,
            numeric_only: bool_t | None = None, **kwargs) -> Series | float:
        return _to_pandas(self._frame).std(axis, skipna, level, ddof, numeric_only, **kwargs)

    def _stat_function(self, name: str, func, axis: Axis | None | lib.NoDefault = None, skipna: bool_t = True,
                       level: Level | None = None, numeric_only: bool_t | None = None, **kwargs):
        return _to_pandas(self._frame)._stat_function(name, func, axis, skipna, level, numeric_only, **kwargs)

    def min(self, axis: Axis | None | lib.NoDefault = lib.no_default, skipna: bool_t = True, level: Level | None = None,
            numeric_only: bool_t | None = None, **kwargs):
        return _to_pandas(self._frame).min(axis, skipna, level, numeric_only, **kwargs)

    def max(self, axis: Axis | None | lib.NoDefault = lib.no_default, skipna: bool_t = True, level: Level | None = None,
            numeric_only: bool_t | None = None, **kwargs):
        return _to_pandas(self._frame).max(axis, skipna, level, numeric_only, **kwargs)

    def mean(self, axis: Axis | None | lib.NoDefault = lib.no_default, skipna: bool_t = True,
             level: Level | None = None, numeric_only: bool_t | None = None, **kwargs) -> Series | float:
        return _to_pandas(self._frame).mean(axis, skipna, level, numeric_only, **kwargs)

    def median(self, axis: Axis | None | lib.NoDefault = lib.no_default, skipna: bool_t = True,
               level: Level | None = None, numeric_only: bool_t | None = None, **kwargs) -> Series | float:
        return _to_pandas(self._frame).median(axis, skipna, level, numeric_only, **kwargs)

    def skew(self, axis: Axis | None | lib.NoDefault = lib.no_default, skipna: bool_t = True,
             level: Level | None = None, numeric_only: bool_t | None = None, **kwargs) -> Series | float:
        return _to_pandas(self._frame).skew(axis, skipna, level, numeric_only, **kwargs)

    def kurt(self, axis: Axis | None | lib.NoDefault = lib.no_default, skipna: bool_t = True,
             level: Level | None = None, numeric_only: bool_t | None = None, **kwargs) -> Series | float:
        return _to_pandas(self._frame).kurt(axis, skipna, level, numeric_only, **kwargs)

    def _min_count_stat_function(self, name: str, func, axis: Axis | None = None, skipna: bool_t = True,
                                 level: Level | None = None, numeric_only: bool_t | None = None, min_count: int = 0,
                                 **kwargs):
        return _to_pandas(self._frame)._min_count_stat_function(name, func, axis, skipna, level, numeric_only,
                                                                min_count, **kwargs)

    def sum(self, axis: Axis | None = None, skipna: bool_t = True, level: Level | None = None,
            numeric_only: bool_t | None = None, min_count=0, **kwargs):
        return _to_pandas(self._frame).sum(axis, skipna, level, numeric_only, min_count, **kwargs)

    def prod(self, axis: Axis | None = None, skipna: bool_t = True, level: Level | None = None,
             numeric_only: bool_t | None = None, min_count: int = 0, **kwargs):
        return _to_pandas(self._frame).prod(axis, skipna, level, numeric_only, min_count, **kwargs)

    def mad(self, axis: Axis | None = None, skipna: bool_t = True, level: Level | None = None) -> Series | float:
        return _to_pandas(self._frame).mad(axis, skipna, level)

    @classmethod
    def _add_numeric_operations(cls):
        _to_pandas(self._frame)._add_numeric_operations()

    def rolling(self, window: int | timedelta | BaseOffset | BaseIndexer, min_periods: int | None = None,
                center: bool_t = False, win_type: str | None = None, on: str | None = None, axis: Axis = 0,
                closed: str | None = None, step: int | None = None, method: str = "single") -> Window | Rolling:
        return _to_pandas(self._frame).rolling(window, min_periods, center, win_type, on, axis, closed, step, method)

    def expanding(self, min_periods: int = 1, center: bool_t | None = None, axis: Axis = 0,
                  method: str = "single") -> Expanding:
        return _to_pandas(self._frame).expanding(min_periods, center, axis, method)

    def ewm(self, com: float | None = None, span: float | None = None,
            halflife: float | TimedeltaConvertibleTypes | None = None, alpha: float | None = None,
            min_periods: int | None = 0, adjust: bool_t = True, ignore_na: bool_t = False, axis: Axis = 0,
            times: str | np.ndarray | pd.DataFrame | Series | None = None,
            method: str = "single") -> ExponentialMovingWindow:
        return _to_pandas(self._frame).ewm(com, span, halflife, alpha, min_periods, adjust, ignore_na, axis, times,
                                           method)

    def _inplace_method(self, other, op):
        return _to_pandas(self._frame)._inplace_method(other, op)

    def __iadd__(self: NDFrameT, other) -> NDFrameT:
        return _to_pandas(self._frame).__iadd__(other)

    def __isub__(self: NDFrameT, other) -> NDFrameT:
        return _to_pandas(self._frame).__isub__(other)

    def __imul__(self: NDFrameT, other) -> NDFrameT:
        return _to_pandas(self._frame).__imul__(other)

    def __itruediv__(self: NDFrameT, other) -> NDFrameT:
        return _to_pandas(self._frame).__itruediv__(other)

    def __ifloordiv__(self: NDFrameT, other) -> NDFrameT:
        return _to_pandas(self._frame).__ifloordiv__(other)

    def __imod__(self: NDFrameT, other) -> NDFrameT:
        return _to_pandas(self._frame).__imod__(other)

    def __ipow__(self: NDFrameT, other) -> NDFrameT:
        return _to_pandas(self._frame).__ipow__(other)

    def __iand__(self: NDFrameT, other) -> NDFrameT:
        return _to_pandas(self._frame).__iand__(other)

    def __ior__(self: NDFrameT, other) -> NDFrameT:
        return _to_pandas(self._frame).__ior__(other)

    def __ixor__(self: NDFrameT, other) -> NDFrameT:
        return _to_pandas(self._frame).__ixor__(other)

    def _find_valid_index(self, *, how: str) -> Hashable | None:
        return _to_pandas(self._frame)._find_valid_index(how=how)

    def first_valid_index(self) -> Hashable | None:
        return _to_pandas(self._frame).first_valid_index()

    def last_valid_index(self) -> Hashable | None:
        return _to_pandas(self._frame).last_valid_index()

    def _reset_cache(self, key: str | None = None) -> None:
        _to_pandas(self._frame)._reset_cache(key)

    def __sizeof__(self) -> int:
        return _to_pandas(self._frame).__sizeof__()

    def _dir_deletions(self) -> set[str]:
        return _to_pandas(self._frame)._dir_deletions()

    def __dir__(self) -> list[str]:
        return _to_pandas(self._frame).__dir__()

    @property
    def iloc(self) -> _iLocIndexer:
        return _to_pandas(self._frame).iloc

    @property
    def loc(self) -> _LocIndexer:
        return _to_pandas(self._frame).loc

    @property
    def at(self) -> _AtIndexer:
        return _to_pandas(self._frame).at

    @property
    def iat(self) -> _iAtIndexer:
        return _to_pandas(self._frame).iat

    def __eq__(self, other):
        return _to_pandas(self._frame).__eq__(other)

    def __ne__(self, other):
        return _to_pandas(self._frame).__ne__(other)

    def __lt__(self, other):
        return _to_pandas(self._frame).__lt__(other)

    def __le__(self, other):
        return _to_pandas(self._frame).__le__(other)

    def __gt__(self, other):
        return _to_pandas(self._frame).__gt__(other)

    def __ge__(self, other):
        return _to_pandas(self._frame).__ge__(other)

    def _logical_method(self, other, op):
        return _to_pandas(self._frame)._logical_method(other, op)

    def __and__(self, other):
        return _to_pandas(self._frame).__and__(other)

    def __rand__(self, other):
        return _to_pandas(self._frame).__rand__(other)

    def __or__(self, other):
        return _to_pandas(self._frame).__or__(other)

    def __ror__(self, other):
        return _to_pandas(self._frame).__ror__(other)

    def __xor__(self, other):
        return _to_pandas(self._frame).__xor__(other)

    def __rxor__(self, other):
        return _to_pandas(self._frame).__rxor__(other)

    def __add__(self, other):
        return _to_pandas(self._frame).__add__(other)

    def __radd__(self, other):
        return _to_pandas(self._frame).__radd__(other)

    def __sub__(self, other):
        return _to_pandas(self._frame).__sub__(other)

    def __rsub__(self, other):
        return _to_pandas(self._frame).__rsub__(other)

    def __mul__(self, other):
        return _to_pandas(self._frame).__mul__(other)

    def __rmul__(self, other):
        return _to_pandas(self._frame).__rmul__(other)

    def __truediv__(self, other):
        return _to_pandas(self._frame).__truediv__(other)

    def __rtruediv__(self, other):
        return _to_pandas(self._frame).__rtruediv__(other)

    def __floordiv__(self, other):
        return _to_pandas(self._frame).__floordiv__(other)

    def __rfloordiv__(self, other):
        return _to_pandas(self._frame).__rfloordiv__(other)

    def __mod__(self, other):
        return _to_pandas(self._frame).__mod__(other)

    def __rmod__(self, other):
        return _to_pandas(self._frame).__rmod__(other)

    def __pow__(self, other):
        return _to_pandas(self._frame).__pow__(other)

    def __rpow__(self, other):
        return _to_pandas(self._frame).__rpow__(other)

    @property
    def __class__(self):
        return _to_pandas(self._frame).__class__

    def __new__(cls):
        return _to_pandas(self._frame).__new__(cls)

    def __str__(self) -> str:
        return _to_pandas(self._frame).__str__()

    def __hash__(self) -> int:
        return _to_pandas(self._frame).__hash__()

    def __format__(self, format_spec: str) -> str:
        return _to_pandas(self._frame).__format__(format_spec)

    def __getattribute__(self, name: str) -> Any:
        return _to_pandas(self._frame).__getattribute__(name)

    def __delattr__(self, name: str) -> None:
        _to_pandas(self._frame).__delattr__(name)

    def __reduce__(self) -> str | Tuple[Any, ...]:
        return _to_pandas(self._frame).__reduce__()

    def __reduce_ex__(self, protocol: SupportsIndex) -> str | Tuple[Any, ...]:
        return _to_pandas(self._frame).__reduce_ex__(protocol)

    def __init_subclass__(self) -> None:
        _to_pandas(self._frame).__init_subclass__()
