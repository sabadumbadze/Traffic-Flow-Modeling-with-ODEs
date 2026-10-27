"""
Dataset Loader for Traffic Flow Forecasting Dataset
===================================================
Provides pure Python loading of 'data/traffic_dataset.mat' without requiring scipy/numpy.
"""

import struct
import zlib

TYPE_MAP = {
    1: ('b', 1),   # miINT8
    2: ('B', 1),   # miUINT8
    3: ('h', 2),   # miINT16
    4: ('H', 2),   # miUINT16
    5: ('i', 4),   # miINT32
    6: ('I', 4),   # miUINT32
    7: ('f', 4),   # miSINGLE
    9: ('d', 8),   # miDOUBLE
    12: ('q', 8),  # miINT64
    13: ('Q', 8),  # miUINT64
}

def parse_tag(buf, offset):
    w1, w2 = struct.unpack('<II', buf[offset:offset+8])
    if w1 > 0xFFFF:
        nbytes = w1 >> 16
        dtype = w1 & 0xFFFF
        val_bytes = buf[offset+4:offset+4+nbytes]
        pad = (4 - (nbytes % 4)) % 4
        next_offset = offset + 4 + nbytes + pad
        return dtype, nbytes, val_bytes, next_offset
    else:
        dtype = w1
        nbytes = w2
        pad = (8 - (nbytes % 8)) % 8
        val_bytes = buf[offset+8:offset+8+nbytes]
        next_offset = offset + 8 + nbytes + pad
        return dtype, nbytes, val_bytes, next_offset

def parse_matrix_header(buf, offset):
    m_type, m_len, m_val, next_off = parse_tag(buf, offset)
    assert m_type == 14, f"Expected miMATRIX (14), got {m_type}"
    f_type, f_len, f_val, sub_off = parse_tag(m_val, 0)
    arr_class = f_val[0]
    flags = f_val[1]
    nzmax = struct.unpack('<I', f_val[4:8])[0]
    
    d_type, d_len, d_val, sub_off = parse_tag(m_val, sub_off)
    dims = struct.unpack(f'<{d_len//4}i', d_val)
    
    n_type, n_len, n_val, sub_off = parse_tag(m_val, sub_off)
    name = n_val.decode('ascii', errors='ignore')
    return arr_class, flags, nzmax, dims, name, m_val, sub_off, next_off

def parse_dense_numeric(buf, offset, dims):
    dt, db, dbytes, _ = parse_tag(buf, offset)
    numel = 1
    for d in dims:
        numel *= d
    fmt_char, size = TYPE_MAP[dt]
    vals = struct.unpack(f'<{numel}{fmt_char}', dbytes[:numel*size])
    rows, cols = dims[0], dims[1]
    return [[vals[r + c * rows] for c in range(cols)] for r in range(rows)]

def parse_sparse_matrix(buf, offset, dims):
    ir_t, ir_len, ir_val, off = parse_tag(buf, offset)
    ir = struct.unpack(f'<{ir_len//4}i', ir_val)
    
    jc_t, jc_len, jc_val, off = parse_tag(buf, off)
    jc = struct.unpack(f'<{jc_len//4}i', jc_val)
    
    pr_t, pr_len, pr_val, off = parse_tag(buf, off)
    fmt_char, size = TYPE_MAP[pr_t]
    n_vals = len(pr_val) // size
    pr = struct.unpack(f'<{n_vals}{fmt_char}', pr_val[:n_vals*size])
    
    rows, cols = dims[0], dims[1]
    mat = [[0.0 for _ in range(cols)] for _ in range(rows)]
    for c in range(cols):
        start_idx = jc[c]
        end_idx = jc[c+1]
        for idx in range(start_idx, end_idx):
            mat[ir[idx]][c] = pr[idx]
    return mat

def load_traffic_dataset(filepath='data/traffic_dataset.mat'):
    """Loads all variables from the MATLAB Level 5 .mat file.

    Returns dict with keys:
      'tra_X_tr': list of 1261 matrices [36 x 48]
      'tra_X_te': list of 840 matrices [36 x 48]
      'tra_Y_tr': matrix [36 x 1261]
      'tra_Y_te': matrix [36 x 840]
      'tra_adj_mat': matrix [36 x 36]
    """
    data = {}
    with open(filepath, 'rb') as f:
        f.read(128) # skip header
        while True:
            tag = f.read(8)
            if len(tag) < 8:
                break
            dtype, nbytes = struct.unpack('<II', tag)
            if dtype != 15:
                break
            raw_comp = f.read(nbytes)
            decomp = zlib.decompress(raw_comp)
            arr_class, flags, nzmax, dims, name, m_val, sub_off, _ = parse_matrix_header(decomp, 0)
            
            if arr_class == 6: # dense numeric
                data[name] = parse_dense_numeric(m_val, sub_off, dims)
            elif arr_class == 1: # cell array
                cells = []
                cur_off = sub_off
                n_cells = dims[0] * dims[1]
                for _ in range(n_cells):
                    c_class, _, _, c_dims, _, c_val, c_sub_off, next_off = parse_matrix_header(m_val, cur_off)
                    if c_class == 5:
                        cells.append(parse_sparse_matrix(c_val, c_sub_off, c_dims))
                    elif c_class == 6:
                        cells.append(parse_dense_numeric(c_val, c_sub_off, c_dims))
                    cur_off = next_off
                data[name] = cells
    return data

if __name__ == '__main__':
    data = load_traffic_dataset()
    print("Dataset successfully loaded:")
    for k, v in data.items():
        if isinstance(v, list) and isinstance(v[0], list) and isinstance(v[0][0], list):
            print(f"  {k}: {len(v)} steps x {len(v[0])} nodes x {len(v[0][0])} features")
        else:
            print(f"  {k}: {len(v)} x {len(v[0])}")
