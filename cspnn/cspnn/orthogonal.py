import torch
import torch.nn as nn
import numpy as np


def create_diag_(A, diag):
    n = A.size(0)
    diag_z = torch.zeros(n - 1)
    diag_z[::2] = diag
    A_init = torch.diag(diag_z, diagonal=1)
    A_init = A_init - A_init.T
    with torch.no_grad():
        A.copy_(A_init)
        return A


def cayley_init_(A):
    size = A.size(0) // 2
    diag = A.new(size).uniform_(0.0, np.pi / 2.0)
    diag = -torch.sqrt((1.0 - torch.cos(diag)) / (1.0 + torch.cos(diag)))
    return create_diag_(A, diag)


def cayley_map(X):
    n = X.size(0)
    Id = torch.eye(n, dtype=X.dtype, device=X.device)
    return torch.linalg.solve(Id - X, Id + X)


class Parametrization(nn.Module):
    """
    Implements the parametrization of a manifold in terms of a Euclidean space

    It gives the parametrized matrix through the attribute `B`

    To use it, subclass it and implement the method `retraction` and the method `forward`
    (and optionally `project`). See the documentation in these methods for details

    You can find an example in the file `orthogonal.py` where we implement the Orthogonal
    class to optimize over the Stiefel manifold using an arbitrary retraction
    """

    def __init__(self, A, base, mode):
        """
        mode: "static" or a tuple such that:
                mode[0] == "dynamic"
                mode[1]: int, K, the number of steps after which we should change
                            the basis of the dyn triv
                mode[2]: int, M, the number of changes of basis after which we
                            should project back onto the manifold the basis.
                            This is particularly helpful for small values of K.
        """
        super(Parametrization, self).__init__()
        assert mode == "static" or (
            isinstance(mode, tuple) and len(mode) == 3 and mode[0] == "dynamic"
        )

        self.A = nn.Parameter(A)
        self.register_buffer("_B", None)
        self.register_buffer("base", base)
        # This is necessary, as it will be generated again the first time that self.B is called
        # We still need to register the buffer though

        if mode == "static":
            self.mode = mode
        else:
            self.mode = mode[0]
            self.K = mode[1]
            self.M = mode[2]
            self.k = 0
            self.m = 0

        # This implements the parametrization trick in a rather slick way.
        # We put a hook on A, such that, whenever its gradients are computed, we
        #  get rid of self._B so that it has to be recomputed the next time that
        #  self.B is accessed
        def hook(grad):
            nonlocal self
            self._B = None

        self.A.register_hook(hook)

    def rebase(self):
        with torch.no_grad():
            self.base.data.copy_(self._B.data)
            self.A.data.zero_()

    # seems we can't use python properties with pytorch modules for now
    # see this issue: https://github.com/pytorch/pytorch/issues/49726
    # @property
    def B(self):
        not_B = self._B is None
        if not_B or (not self._B.grad_fn and torch.is_grad_enabled()):
            self._B = self.retraction(self.A, self.base)
            # Just to be safe
            self._B.requires_grad_()
            # Now self._B it's not a leaf tensor, so we convert it into a leaf
            self._B.retain_grad()

            # Increment the counters for the dyntriv algorithm if we have generated B
            if self.mode == "dynamic" and not_B:
                if self.k == 0:
                    self.rebase()
                    # Project the base back to the manifold every M changes of base
                    # Increment the counter before as we don't project the first time
                    self.m = (self.m + 1) % self.M
                    # It's optional to implement this method
                    if self.m == 0 and hasattr(self, "project"):
                        with torch.no_grad():
                            self.base = self.project(self.base)
                # Change the basis after K optimization steps
                # Increment the counter afterwards as we change the basis in the first iteration
                if self.K != "infty":
                    self.k = (self.k + 1) % self.K
                else:
                    # Make sure that we just update the base once
                    if self.k == 0:
                        self.k = 1

        return self._B

    def retraction(self, A, base):
        """
        It computes r_{base}(A).
        Notice that A will not always be in the tangent space of our manifold
          For this reason, we first have to use A to parametrize the tangent space,
          and then compute the retraction
        When dealing with Lie groups, raw_A is always projected into the Lie algebra,
          as an optimization (cf. Section E in the paper)

        more:
        -----
        A retraction retr_p can be interpreted as a first order
        approximation to the exponential map exp_p.
            see: chapter 4.1 of Optomization algorithms on matrix manifolds
        """
        raise NotImplementedError

    def project(self, base):
        """
        This method is OPTIONAL
        It returns the projected base back into the manifold
        """
        raise NotImplementedError

    def forward(self, input):
        """
        It uses the attribute self.B to implement the layer itself (e.g. Linear, CNN, ...)
        """
        raise NotImplementedError


class Orthogonal(Parametrization):
    """Class that implements optimization restricted to the Stiefel manifold"""

    def __init__(self, input_size, output_size, initializer_skew, mode, param):
        """
        mode: "static" or a tuple such that:
                mode[0] == "dynamic"
                mode[1]: int, K, the number of steps after which we should change the basis of the dyn triv
                mode[2]: int, M, the number of changes of basis after which we should project back onto the manifold the basis. This is particularly helpful for small values of K.

        param: A parametrization of in terms of skew-symmetyric matrices
        """
        max_size = max(input_size, output_size)
        A = torch.empty(max_size, max_size)
        base = torch.empty(max_size, max_size)
        super(Orthogonal, self).__init__(A, base, mode)
        self.input_size = input_size
        self.output_size = output_size
        self.param = param
        self.init_A = initializer_skew
        self.init_base = nn.init.eye_

        self.reset_parameters()

    def reset_parameters(self):
        self.init_A(self.A)
        self.init_base(self.base)

    def retraction(self, A, base):
        # This could be any parametrization of a tangent space
        A = A.triu(diagonal=1)
        A = A - A.t()
        B = base.mm(self.param(A))
        if self.input_size != self.output_size:
            B = B[: self.input_size, : self.output_size]
        return B

    def project(self, base):
        try:
            # Compute the projection using the thin SVD decomposition
            U, _, V = torch.svd(base, some=True)
            return U.mm(V.t())
        except RuntimeError:
            # If the svd does not converge, fallback to the (thin) QR decomposition
            x = base
            if base.size(0) < base.size(1):
                x = base.t()
            ret = torch.qr(x, some=True).Q
            if base.size(0) < base.size(1):
                ret = ret.t()
            return ret
