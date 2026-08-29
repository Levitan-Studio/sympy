"""An explicit matrix that does not evaluate under arithmetic."""

from sympy.core.basic import Basic
from sympy.core.decorators import call_highest_priority
from sympy.core.evalf import prec_to_dps
from sympy.core.numbers import Integer
from sympy.core.singleton import S
from sympy.matrices.expressions.determinant import Determinant
from sympy.matrices.expressions.matadd import MatAdd
from sympy.matrices.expressions.matexpr import MatrixExpr, _sympifyit
from sympy.matrices.expressions.matmul import MatMul
from sympy.matrices.expressions.matpow import MatPow
from sympy.matrices.immutable import ImmutableDenseMatrix


class UnevaluatedMatrix(MatrixExpr):
    """An explicit matrix whose arithmetic stays unevaluated.

    ``UnevaluatedMatrix`` holds an :class:`~.ImmutableDenseMatrix` and knows
    all of its entries, so it can be indexed and printed like an explicit
    matrix.  Unlike an explicit matrix it does not compute anything when it
    takes part in an operation: ``+``, ``-``, ``*`` and ``**`` build
    unevaluated :class:`~.MatAdd`, :class:`~.MatMul` and :class:`~.MatPow`
    nodes, and ``.T``, ``.det()`` and ``.inv()`` build the corresponding
    symbolic nodes.  Calling :meth:`doit` collapses the whole expression to
    an ordinary immutable matrix.

    This makes it possible to show a matrix computation before its result,
    which is what ``Matrix`` itself cannot do.

    Examples
    ========

    >>> from sympy.matrices.matrices_uneval import UnevaluatedMatrix
    >>> A = UnevaluatedMatrix([[1, 2], [3, 4]])
    >>> B = UnevaluatedMatrix([[1, 5], [2, 1]])

    Arithmetic builds an expression tree instead of a result:

    >>> (A + B).is_MatAdd
    True
    >>> (A + B).doit()
    Matrix([
    [2, 7],
    [5, 5]])
    >>> (A*B).doit()
    Matrix([
    [ 5,  7],
    [11, 19]])
    >>> (A - B).doit()
    Matrix([
    [0, -3],
    [1,  3]])
    >>> (A**2).doit()
    Matrix([
    [ 7, 10],
    [15, 22]])

    The entries are still available, so the object behaves like a matrix:

    >>> A.shape
    (2, 2)
    >>> A[0, 1]
    2
    >>> A.T.doit()
    Matrix([
    [1, 3],
    [2, 4]])
    >>> A.det().doit()
    -2

    See Also
    ========

    sympy.matrices.immutable.ImmutableDenseMatrix
    sympy.matrices.expressions.matexpr.MatrixExpr
    """

    __slots__ = ()

    # Beat ImmutableDenseMatrix (11.0) so that mixed operations such as
    # ``Matrix(...) + UnevaluatedMatrix(...)`` are left unevaluated too.
    _op_priority = 12.0

    def __new__(cls, *args, **kwargs):
        return Basic.__new__(cls, ImmutableDenseMatrix(*args, **kwargs))

    @property
    def matrix(self):
        """The explicit :class:`~.ImmutableDenseMatrix` being wrapped."""
        return self.args[0]

    @property
    def shape(self):
        rows, cols = self.matrix.shape
        return Integer(rows), Integer(cols)

    def _entry(self, i, j, **kwargs):
        return self.matrix[i, j]

    def doit(self, **hints):
        return self.matrix.doit(**hints)

    def as_explicit(self):
        return self.matrix

    def det(self):
        # ``MatrixExpr.det`` evaluates the determinant; keep it symbolic
        # instead, in line with ``.T`` and ``.inv()``.
        return Determinant(self)

    @_sympifyit('other', NotImplemented)
    @call_highest_priority('__radd__')
    def __add__(self, other):
        return MatAdd(self, other)

    @_sympifyit('other', NotImplemented)
    @call_highest_priority('__add__')
    def __radd__(self, other):
        return MatAdd(other, self)

    @_sympifyit('other', NotImplemented)
    @call_highest_priority('__rsub__')
    def __sub__(self, other):
        return MatAdd(self, -other)

    @_sympifyit('other', NotImplemented)
    @call_highest_priority('__sub__')
    def __rsub__(self, other):
        return MatAdd(other, -self)

    @_sympifyit('other', NotImplemented)
    @call_highest_priority('__rmul__')
    def __mul__(self, other):
        return MatMul(self, other)

    @_sympifyit('other', NotImplemented)
    @call_highest_priority('__mul__')
    def __rmul__(self, other):
        return MatMul(other, self)

    @_sympifyit('other', NotImplemented)
    @call_highest_priority('__rmul__')
    def __matmul__(self, other):
        return MatMul(self, other)

    @_sympifyit('other', NotImplemented)
    @call_highest_priority('__mul__')
    def __rmatmul__(self, other):
        return MatMul(other, self)

    @_sympifyit('other', NotImplemented)
    @call_highest_priority('__rpow__')
    def __pow__(self, other):
        return MatPow(self, other)

    def __neg__(self):
        return MatMul(S.NegativeOne, self)

    def _eval_derivative(self, x):
        return self.func(self.matrix.applyfunc(lambda e: e.diff(x)))

    def _eval_evalf(self, prec):
        return self.func(self.matrix.evalf(n=prec_to_dps(prec)))

    def _sympystr(self, printer, *args):
        return printer._print(self.matrix)

    def _latex(self, printer, *args):
        return printer._print(self.matrix)

    def _pretty(self, printer, *args):
        return printer._print(self.matrix)
