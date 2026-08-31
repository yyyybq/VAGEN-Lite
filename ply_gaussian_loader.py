#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PLY Gaussian Splat Loader for standard and SuperSplat compressed PLY."""
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
from plyfile import PlyData

SH_C0 = 0.28209479177387814

@dataclass
class GaussianSplatData:
    positions: np.ndarray
    rotations: np.ndarray  # WXYZ
    scales: np.ndarray
    colors_dc: np.ndarray
    opacities: np.ndarray
    sh_rest: Optional[np.ndarray] = None
    num_splats: int = 0
    sh_bands: int = 0
    is_compressed: bool = False

    def __post_init__(self):
        self.num_splats = len(self.positions)
        if self.sh_rest is not None:
            coeffs_per_channel = self.sh_rest.shape[1] // 3
            self.sh_bands = {0: 0, 3: 1, 8: 2, 15: 3}.get(coeffs_per_channel, 0)
        else:
            self.sh_bands = 0

    def get_linear_colors(self) -> np.ndarray:
        return np.clip(self.colors_dc * SH_C0 + 0.5, 0.0, 1.0).astype(np.float32)

    def get_rotations_xyzw(self) -> np.ndarray:
        rotations_xyzw = self.rotations[:, [1, 2, 3, 0]]
        norms = np.linalg.norm(rotations_xyzw, axis=1, keepdims=True)
        return (rotations_xyzw / np.maximum(norms, 1e-8)).astype(np.float32)

    def get_sh_coefficients(self) -> Optional[np.ndarray]:
        if self.sh_rest is None:
            return None
        n = self.num_splats
        total = self.sh_rest.shape[1]
        if total % 3 != 0:
            return None
        k = total // 3
        r = self.sh_rest[:, 0:k]
        g = self.sh_rest[:, k:2*k]
        b = self.sh_rest[:, 2*k:3*k]
        sh_rest_reshaped = np.stack([r, g, b], axis=2).astype(np.float32)
        dc_coeffs = self.colors_dc.reshape(n, 1, 3).astype(np.float32)
        return np.concatenate([dc_coeffs, sh_rest_reshaped], axis=1)

class ChunkData:
    def __init__(self, chunk_element):
        self.min_x = np.asarray(chunk_element['min_x'], dtype=np.float32)
        self.min_y = np.asarray(chunk_element['min_y'], dtype=np.float32)
        self.min_z = np.asarray(chunk_element['min_z'], dtype=np.float32)
        self.max_x = np.asarray(chunk_element['max_x'], dtype=np.float32)
        self.max_y = np.asarray(chunk_element['max_y'], dtype=np.float32)
        self.max_z = np.asarray(chunk_element['max_z'], dtype=np.float32)
        self.min_scale_x = np.asarray(chunk_element['min_scale_x'], dtype=np.float32)
        self.min_scale_y = np.asarray(chunk_element['min_scale_y'], dtype=np.float32)
        self.min_scale_z = np.asarray(chunk_element['min_scale_z'], dtype=np.float32)
        self.max_scale_x = np.asarray(chunk_element['max_scale_x'], dtype=np.float32)
        self.max_scale_y = np.asarray(chunk_element['max_scale_y'], dtype=np.float32)
        self.max_scale_z = np.asarray(chunk_element['max_scale_z'], dtype=np.float32)
        self.min_r = np.asarray(chunk_element['min_r'], dtype=np.float32)
        self.min_g = np.asarray(chunk_element['min_g'], dtype=np.float32)
        self.min_b = np.asarray(chunk_element['min_b'], dtype=np.float32)
        self.max_r = np.asarray(chunk_element['max_r'], dtype=np.float32)
        self.max_g = np.asarray(chunk_element['max_g'], dtype=np.float32)
        self.max_b = np.asarray(chunk_element['max_b'], dtype=np.float32)
        self.chunk_count = len(self.min_x)
        self.chunk_size = 256

class PLYGaussianLoader:
    def __init__(self):
        self.chunk_data = None
        self.is_compressed_format = False

    def load_ply(self, ply_path: str) -> GaussianSplatData:
        with open(ply_path, 'rb') as f:
            ply = PlyData.read(f)
        names = [e.name for e in ply.elements]
        self.is_compressed_format = 'chunk' in names and 'vertex' in names
        return self._load_compressed_ply(ply) if self.is_compressed_format else self._load_standard_ply(ply)

    def _load_standard_ply(self, ply: PlyData) -> GaussianSplatData:
        v = ply['vertex']
        n = len(v)
        names = v.data.dtype.names
        positions = np.column_stack([v['x'], v['y'], v['z']]).astype(np.float32)
        if all(p in names for p in ['rot_0', 'rot_1', 'rot_2', 'rot_3']):
            rotations = np.column_stack([v['rot_0'], v['rot_1'], v['rot_2'], v['rot_3']]).astype(np.float32)
        else:
            rotations = np.column_stack([
                v['rot_w'].astype(np.float32) if 'rot_w' in names else np.ones(n, dtype=np.float32),
                v['rot_x'].astype(np.float32) if 'rot_x' in names else np.zeros(n, dtype=np.float32),
                v['rot_y'].astype(np.float32) if 'rot_y' in names else np.zeros(n, dtype=np.float32),
                v['rot_z'].astype(np.float32) if 'rot_z' in names else np.zeros(n, dtype=np.float32),
            ])
        rotations /= np.maximum(np.linalg.norm(rotations, axis=1, keepdims=True), 1e-8)
        scales = np.column_stack([v['scale_0'], v['scale_1'], v['scale_2']]).astype(np.float32)
        colors_dc = np.column_stack([v['f_dc_0'], v['f_dc_1'], v['f_dc_2']]).astype(np.float32)
        opacities = np.asarray(v['opacity'], dtype=np.float32).reshape(-1, 1)
        sh_cols = [f'f_rest_{i}' for i in range(45) if f'f_rest_{i}' in names]
        sh_rest = np.column_stack([v[c] for c in sh_cols]).astype(np.float32) if sh_cols else None
        return GaussianSplatData(positions, rotations, scales, colors_dc, opacities, sh_rest, is_compressed=False)

    def _load_compressed_ply(self, ply: PlyData) -> GaussianSplatData:
        self.chunk_data = ChunkData(ply['chunk'])
        v = ply['vertex']
        n = len(v)
        ci = self._compute_chunk_indices_sequential(n)
        positions = self._unpack_position_with_indices(np.asarray(v['packed_position'], dtype=np.uint32), ci)
        scales = self._unpack_scale_with_indices(np.asarray(v['packed_scale'], dtype=np.uint32), ci)
        rotations = self._unpack_rotation_supersplat(np.asarray(v['packed_rotation'], dtype=np.uint32))
        colors_dc, opacities = self._unpack_color(np.asarray(v['packed_color'], dtype=np.uint32), ci)
        sh_rest = self._unpack_sh_coefficients(ply['sh']) if 'sh' in [e.name for e in ply.elements] else None
        return GaussianSplatData(positions, rotations, scales, colors_dc, opacities, sh_rest, is_compressed=True)

    def _compute_chunk_indices_sequential(self, n: int) -> np.ndarray:
        idx = (np.arange(n, dtype=np.int64) // self.chunk_data.chunk_size).astype(np.int32)
        np.clip(idx, 0, self.chunk_data.chunk_count - 1, out=idx)
        return idx

    def _unpack_position_with_indices(self, packed: np.ndarray, ci: np.ndarray) -> np.ndarray:
        xq = ((packed >> 21) & 0x7FF).astype(np.float32) / 2047.0
        yq = ((packed >> 11) & 0x3FF).astype(np.float32) / 1023.0
        zq = (packed & 0x7FF).astype(np.float32) / 2047.0
        out = np.empty((packed.shape[0], 3), dtype=np.float32)
        cd = self.chunk_data
        for c in np.unique(ci):
            m = ci == c; c = int(c)
            out[m,0] = cd.min_x[c] + xq[m] * (cd.max_x[c] - cd.min_x[c])
            out[m,1] = cd.min_y[c] + yq[m] * (cd.max_y[c] - cd.min_y[c])
            out[m,2] = cd.min_z[c] + zq[m] * (cd.max_z[c] - cd.min_z[c])
        return out

    def _unpack_scale_with_indices(self, packed: np.ndarray, ci: np.ndarray) -> np.ndarray:
        sxq = ((packed >> 21) & 0x7FF).astype(np.float32) / 2047.0
        syq = ((packed >> 11) & 0x3FF).astype(np.float32) / 1023.0
        szq = (packed & 0x7FF).astype(np.float32) / 2047.0
        out = np.empty((packed.shape[0], 3), dtype=np.float32)
        cd = self.chunk_data
        for c in np.unique(ci):
            m = ci == c; c = int(c)
            out[m,0] = cd.min_scale_x[c] + sxq[m] * (cd.max_scale_x[c] - cd.min_scale_x[c])
            out[m,1] = cd.min_scale_y[c] + syq[m] * (cd.max_scale_y[c] - cd.min_scale_y[c])
            out[m,2] = cd.min_scale_z[c] + szq[m] * (cd.max_scale_z[c] - cd.min_scale_z[c])
        return out

    def _unpack_rotation_supersplat(self, packed: np.ndarray) -> np.ndarray:
        pr = packed.astype(np.uint32)
        largest = (pr >> 30) & 0x3
        bits = pr & 0x3FFFFFFF
        c2 = (bits >> 20) & 0x3FF
        c1 = (bits >> 10) & 0x3FF
        c0 = bits & 0x3FF
        def unpack(v):
            return ((v.astype(np.float32) / 1023.0) * 2.0 - 1.0) * (1.0 / np.sqrt(2.0))
        v0, v1, v2 = unpack(c0), unpack(c1), unpack(c2)
        q = np.zeros((pr.shape[0], 4), dtype=np.float32)  # WXYZ
        xyzw_to_wxyz = [1, 2, 3, 0]
        for li in range(4):
            m = largest == li
            if not np.any(m):
                continue
            stored = [i for i in range(4) if i != li]
            i0, i1, i2 = (xyzw_to_wxyz[stored[0]], xyzw_to_wxyz[stored[1]], xyzw_to_wxyz[stored[2]])
            q[m, i0] = v2[m]; q[m, i1] = v1[m]; q[m, i2] = v0[m]
            qi = xyzw_to_wxyz[li]
            s = 1.0 - (q[m, i0]**2 + q[m, i1]**2 + q[m, i2]**2)
            q[m, qi] = np.sqrt(np.clip(s, 0.0, 1.0))
        q /= np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-8)
        return q

    def _unpack_color(self, packed: np.ndarray, ci: np.ndarray, order: str = 'RGBA', return_logit: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        val = packed.astype(np.uint32).reshape(-1)
        shifts = {'RGBA': (24,16,8,0), 'BGRA': (8,16,24,0), 'ARGB': (16,8,0,24), 'ABGR': (0,8,16,24)}[order]
        r8 = ((val >> shifts[0]) & 0xFF).astype(np.float32)
        g8 = ((val >> shifts[1]) & 0xFF).astype(np.float32)
        b8 = ((val >> shifts[2]) & 0xFF).astype(np.float32)
        a8 = ((val >> shifts[3]) & 0xFF).astype(np.float32)
        ci = np.clip(ci, 0, self.chunk_data.chunk_count - 1)
        colors_dc = np.stack([
            self.chunk_data.min_r[ci] + (r8 / 255.0) * (self.chunk_data.max_r[ci] - self.chunk_data.min_r[ci]),
            self.chunk_data.min_g[ci] + (g8 / 255.0) * (self.chunk_data.max_g[ci] - self.chunk_data.min_g[ci]),
            self.chunk_data.min_b[ci] + (b8 / 255.0) * (self.chunk_data.max_b[ci] - self.chunk_data.min_b[ci]),
        ], axis=1).astype(np.float32)
        colors_dc = (colors_dc - 0.5) / SH_C0
        alpha = (a8 / 255.0).reshape(-1, 1).astype(np.float32)
        if return_logit:
            a = np.clip(alpha, 1e-6, 1.0 - 1e-6)
            alpha = (-np.log(1.0 / a - 1.0)).astype(np.float32)
        return colors_dc, alpha

    def _unpack_sh_coefficients(self, sh_element) -> Optional[np.ndarray]:
        names = sh_element.data.dtype.names
        cols = []
        i = 0
        while f'f_rest_{i}' in names:
            cols.append(f'f_rest_{i}')
            i += 1
        if not cols:
            return None
        sh_u8 = np.column_stack([np.asarray(sh_element[name], dtype=np.uint8) for name in cols])
        return ((sh_u8.astype(np.float32) - 127.5) / 32.0).astype(np.float32)
