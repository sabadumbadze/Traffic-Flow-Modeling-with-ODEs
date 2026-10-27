"""Pure-Python reader for MATLAB level-5 (.mat) files.

Only what the traffic dataset needs is implemented: dense numeric matrices,
sparse matrices (returned as dense lists of rows) and cell arrays, with or
without zlib compression. No third-party library is required.

Return values
-------------
* dense / sparse matrix -> list of rows, each row a list of floats
* cell array            -> list of the (recursively parsed) cell contents
* unsupported classes   -> None
"""

from __future__ import annotations

import struct
import zlib
from typing import Any, Dict, Iterable, List, Optional, Tuple

# MAT data-type codes -> (struct format char, size in bytes)
_TYPE_FORMATS = {
    1: ("b", 1),    # miINT8
    2: ("B", 1),    # miUINT8
    3: ("h", 2),    # miINT16
    4: ("H", 2),    # miUINT16
    5: ("i", 4),    # miINT32
    6: ("I", 4),    # miUINT32
    7: ("f", 4),    # miSINGLE
    9: ("d", 8),    # miDOUBLE
    12: ("q", 8),   # miINT64
    13: ("Q", 8),   # miUINT64
}
_MI_MATRIX = 14
_MI_COMPRESSED = 15
_CLASS_CELL = 1
_CLASS_SPARSE = 5
_DENSE_CLASSES = frozenset(range(6, 16))  # double, single, (u)int8 ... (u)int64
_COMPLEX_FLAG = 0x08


def _read_tag(buf: bytes, offset: int) -> Tuple[int, int, bytes, int]:
    """Read one data element. Returns (data_type, n_bytes, payload, next_offset)."""
    word1, word2 = struct.unpack_from("<II", buf, offset)
    if word1 > 0xFFFF:  # "small data element": type and size share the first word
        n_bytes = word1 >> 16
        data_type = word1 & 0xFFFF
        return data_type, n_bytes, bytes(buf[offset + 4: offset + 4 + n_bytes]), offset + 8
    n_bytes = word2
    payload = bytes(buf[offset + 8: offset + 8 + n_bytes])
    padded = ((n_bytes + 7) // 8) * 8
    return word1, n_bytes, payload, offset + 8 + padded


def _decode_numbers(data_type: int, raw: bytes) -> Tuple[Any, ...]:
    try:
        fmt, size = _TYPE_FORMATS[data_type]
    except KeyError as exc:
        raise NotImplementedError(f"Unsupported MAT numeric type code {data_type}") from exc
    count = len(raw) // size
    return struct.unpack("<%d%s" % (count, fmt), raw[: count * size])


def _to_rows(values: Tuple[Any, ...], dims: Tuple[int, ...]) -> List[List[float]]:
    """MATLAB stores matrices column-major; return a list of rows."""
    if len(dims) != 2:
        raise NotImplementedError(f"Only 2-D arrays are supported, got dims={dims}")
    rows, cols = dims
    return [[float(values[r + c * rows]) for c in range(cols)] for r in range(rows)]


def _read_header(body: bytes) -> Tuple[int, int, Tuple[int, ...], str, int]:
    """Parse array flags, dimensions and name. Returns (class, flags, dims, name, next_pos)."""
    _, _, flags_raw, pos = _read_tag(body, 0)
    array_class = flags_raw[0]
    flags = flags_raw[1]
    _, _, dims_raw, pos = _read_tag(body, pos)
    dims = struct.unpack("<%di" % (len(dims_raw) // 4), dims_raw)
    _, _, name_raw, pos = _read_tag(body, pos)
    return array_class, flags, dims, name_raw.decode("ascii", errors="ignore"), pos


def _count(dims: Tuple[int, ...]) -> int:
    total = 1
    for d in dims:
        total *= d
    return total


def _read_payload(
    body: bytes,
    array_class: int,
    flags: int,
    dims: Tuple[int, ...],
    pos: int,
    max_cells: Optional[int],
) -> Any:
    if flags & _COMPLEX_FLAG:
        raise NotImplementedError("Complex arrays are not supported")

    if array_class == _CLASS_CELL:
        items: List[Any] = []
        for _ in range(_count(dims)):
            if max_cells is not None and len(items) >= max_cells:
                break
            _, value, pos = _read_matrix(body, pos)
            items.append(value)
        return items

    if array_class == _CLASS_SPARSE:
        ir_type, _, ir_raw, pos = _read_tag(body, pos)
        jc_type, _, jc_raw, pos = _read_tag(body, pos)
        pr_type, _, pr_raw, pos = _read_tag(body, pos)
        row_index = _decode_numbers(ir_type, ir_raw)
        col_pointer = _decode_numbers(jc_type, jc_raw)
        values = _decode_numbers(pr_type, pr_raw)
        n_rows, n_cols = dims
        dense = [[0.0] * n_cols for _ in range(n_rows)]
        for c in range(n_cols):
            for idx in range(col_pointer[c], col_pointer[c + 1]):
                dense[row_index[idx]][c] = float(values[idx])
        return dense

    if array_class in _DENSE_CLASSES:
        data_type, _, raw, pos = _read_tag(body, pos)
        return _to_rows(_decode_numbers(data_type, raw), dims)

    return None


def _read_matrix(buf: bytes, offset: int) -> Tuple[str, Any, int]:
    """Parse a (possibly nested) miMATRIX element. Returns (name, value, next_offset)."""
    data_type, _, body, next_offset = _read_tag(buf, offset)
    if data_type != _MI_MATRIX:
        raise ValueError(f"Expected miMATRIX element, found type code {data_type}")
    array_class, flags, dims, name, pos = _read_header(body)
    return name, _read_payload(body, array_class, flags, dims, pos, None), next_offset


def load_mat(
    path: Any,
    names: Optional[Iterable[str]] = None,
    max_cells: Optional[int] = None,
) -> Dict[str, Any]:
    """Load variables from a MATLAB v5 file.

    Parameters
    ----------
    names:
        If given, only variables with these names are parsed.
    max_cells:
        For top-level cell arrays, parse only the first ``max_cells`` cells.
        Handy for peeking at the static features in the first snapshot of the
        1261-cell ``tra_X_tr`` array without decoding all of it.
    """
    wanted = None if names is None else set(names)
    with open(path, "rb") as handle:
        data = handle.read()
    if data[126:128] != b"IM":
        raise NotImplementedError("Only little-endian MATLAB v5 files are supported")

    result: Dict[str, Any] = {}
    pos = 128
    while pos + 8 <= len(data):
        element_type, n_bytes = struct.unpack_from("<II", data, pos)
        if element_type == _MI_COMPRESSED:
            buf = zlib.decompress(data[pos + 8: pos + 8 + n_bytes])
            offset = 0
            pos += 8 + n_bytes
        elif element_type == _MI_MATRIX:
            buf = data
            offset = pos
            pos += 8 + ((n_bytes + 7) // 8) * 8
        else:
            break

        tag_type, _, body, _ = _read_tag(buf, offset)
        if tag_type != _MI_MATRIX:
            continue
        array_class, flags, dims, name, body_pos = _read_header(body)
        if wanted is not None and name not in wanted:
            continue
        result[name] = _read_payload(body, array_class, flags, dims, body_pos, max_cells)
    return result
