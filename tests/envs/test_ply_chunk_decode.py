"""Packed point decoding preserves per-vertex chunk identity and endpoint values."""
from types import SimpleNamespace
import numpy as np
from ply_gaussian_loader import PLYGaussianLoader


def loader():
    value=PLYGaussianLoader()
    fields={}
    for prefix in ('','scale_'):
        for axis,low,high in [('x',[-2,4],[2,8]),('y',[-3,10],[1,12]),('z',[5,-7],[9,-3])]:
            fields['min_'+prefix+axis]=np.array(low,dtype=np.float32)
            fields['max_'+prefix+axis]=np.array(high,dtype=np.float32)
    value.chunk_data=SimpleNamespace(**fields)
    return value


def test_interleaved_chunk_extrema():
    dec=loader();packed=np.array([0,0xffffffff,0xffffffff,0],dtype=np.uint32);chunks=np.array([1,0,1,0])
    expected=np.array([[4,10,-7],[2,1,9],[8,12,-3],[-2,-3,5]],dtype=np.float32)
    for decode in [dec._unpack_position_with_indices,dec._unpack_scale_with_indices]:
        actual=decode(packed,chunks)
        assert actual.dtype==np.float32
        np.testing.assert_array_equal(actual,expected)


def test_axis_bitfields_are_not_swapped():
    dec=loader();packed=np.array([0x7ff<<21,0x3ff<<11,0x7ff],dtype=np.uint32);chunks=np.zeros(3,dtype=np.int32)
    expected=np.array([[2,-3,5],[-2,1,5],[-2,-3,9]],dtype=np.float32)
    for decode in [dec._unpack_position_with_indices,dec._unpack_scale_with_indices]:
        np.testing.assert_array_equal(decode(packed,chunks),expected)
