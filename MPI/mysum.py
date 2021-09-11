import numpy as np
from mpi4py import MPI


def mysum_obj(a, b):
    if not isinstance(a, np.ndarray):
        return []
    na, ca, ha, wa = a.shape
    nb, cb, hb, wb = b.shape
    if ca != cb or ha != hb or wa != wb:
        return []
    c = np.zeros([na+nb, ca, ha, wa])
    for n in range(na):
        c[n] = a[n]
    for n in range(nb):
        c[na+n] = b[n]
    return c


def mysum_buf(a, b, dt):
    assert dt == MPI.INT
    assert len(a) == len(b)

    def to_nyarray(a):
        # convert a MPI.memory object to a numpy array
        size = len(a)
        buf = np.array(a, dtype='B', copy=False)
        return np.ndarray(buffer=buf, dtype='i', shape=(size / 4,))

    to_nyarray(b)[:] = mysum_obj(to_nyarray(a), to_nyarray(b))


def mysum(ba, bb, dt):
    if dt is None:
        # ba, bb are python objects
        return mysum_obj(ba, bb)
    else:
        # ba, bb are MPI.memory objects
        return mysum_buf(ba, bb, dt)
