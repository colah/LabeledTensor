import numpy as np
 

class Axis:
    def __init__(self, name, n):
        self.name = name
        self.n = n

    def __repr__(self):
        return self.name + "~" + repr(self.n)

class NumpyLabeledTensor(object):

    def __init__(self, m, axes):
        m = np.asarray(m)
        assert m.ndim == len(axes)
        self.m = m
        self._axes = axes

    @property
    def axes(self):
        return set(self._axes)

    def _asarray(self, x):
        pass


    def _dict_to_npindex(self, ind_dict):
        np_index = []
        for ax in self._axes:
            if ax in ind_dict:
                np_index.append(ind_dict[ax])
            else:
                np_index.append(slice(None))
        return tuple(np_index)

    def __repr__(self):
        return repr(self.m)

    def __add__(self, y):
        x,y = align(self, y)
        return NumpyLabeledTensor(x.m+y.m, x._axes)

    def __mul__(self, y):
        x,y = align(self, y)
        return NumpyLabeledTensor(x.m*y.m, x._axes)

    def __getitem__(self, inds):
        ind_dict = inds_to_dict(inds)
        np_ind   = self._dict_to_npindex(ind_dict)

        ret_inds = filter(lambda x: x not in ind_dict, self._axes)
        m = self.m[np_ind]
        if ret_inds == []:
            return float(m)
        else:
            return NumpyLabeledTensor(m, ret_inds)

    def __setitem__(self, inds, val):
        ind_dict = inds_to_dict(inds)
        np_ind   = self._dict_to_npindex(ind_dict)
        if type(val) is NumpyLabeledTensor:
            val = val.m
        self.m[np_ind] = val


def align(A, B):
    """Given two arrays A and B with potentially varying axes,
       _align will align the shared axes, and create new
       broadcastable axes for unshared axes."""

    A_axes_set = set(A._axes)
    B_axes_set = set(B._axes)

    AB = list(A_axes_set & B_axes_set)
    Ax = list(A_axes_set - B_axes_set)
    Bx = list(B_axes_set - A_axes_set)

    axes =list(A_axes_set + B_axes_set)

    A_axes = Bx + A._axes
    A_m = A.m[tuple(len(Bx)*[np.newaxis])]
    A_m = A_m.transpose(tuple([A_axes.index(ax) for ax in axes]))

    B_axes = Ax + B._axes
    B_m = B.m[tuple(len(Ax)*[np.newaxis])]
    B_m = B_m.transpose(tuple([B_axes.index(ax) for ax in axes]))

    A_ = NumpyLabeledTensor(A_m, axes)
    B_ = NumpyLabeledTensor(B_m, axes)

    return A_, B_

def dot(A, B, to_sum = None):

    symbs = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXKYZ"
    used_symbs = []
    ax_symbs = {}

    all_axes = A.axes | B.axes

    for ax in all_axes:
        if len(ax.name) == 1 and ax.name not in used_symbs:
            symb = ax.name
        elif len(ax.name) == 1 and ax.name.lower() not in used_symbs:
            symb = ax.name.lower()
        else:
            symb = filter(lambda x: x not in used_symbs, symbs)[0]
        used_symbs.append(symb)
        ax_symbs [ax] = symb
    
    to_sum = to_sum or (A.axes & B.axes)
    out_axes = list(all_axes - to_sum)

    astr = ''.join([ax_symbs[ax] for ax in A._axes])
    bstr = ''.join([ax_symbs[ax] for ax in B._axes])
    mstr = ''.join([ax_symbs[ax] for ax in out_axes])

    fstr = astr + "," + bstr + "->" + mstr
    m = np.einsum(fstr, A.m, B.m)#, bstr, mstr)
    
    return NumpyLabeledTensor(m, out_axes)


def inds_to_dict(inds):
    """For indexing, we hijack the slicing mechanism to emulate,
       indexing with a dictionary. For example, we want to be
       able to do things like:
       
          a[X: 1, Y: 2:3, Z: 3] 

       Unfortunately, a will receive a very messy index:

          (slice(X, 1), slice(Y, 2, 3), slice(Z, 3))

       This function can preprocess an index to convert it into a
       dictionary, and catch invalid indeces. For example, it will
       convert the above to:

           {X: 1, Y: slice(2,3), Z: 3}

        slices disabled for now
       """
    if type(inds) is tuple:
        inds = list(inds)
    else:
        inds = [inds]
    ind_dict = {}
    for ind in inds:
        # We are hijacking slices for
        # our desired notation.
        assert type(ind) is slice
        axis = ind.start
        pos  = ind.stop
        stop  = None #ind.step
        assert ind.step == None #TODO
        if stop:
            ind_dict[axis] = slice(pos, stop)
        else:
            ind_dict[axis] = pos
    return ind_dict



def zeros(*axes):
    axes = list(set(axes))
    m = np.zeros(tuple([ax.n for ax in axes]) )
    return NumpyLabeledTensor(m, axes)

#def randn(*axes):
#    axes = list(set(axes))
#    m = np.zeros(tuple([ax.n for ax in axes]) )
#    return NumpyLabeledTensor(m, axes)



X = Axis("X", 4)
Y = Axis("Y", 3)
Z = Axis("Z", 2)



arr = zeros(X, Y)

arr[Y: 1] = 2
arr[X: 3] = 3

a = zeros(X)
a[X: 2] = 4

b = zeros(Y)
b[Y: 1] = 3

