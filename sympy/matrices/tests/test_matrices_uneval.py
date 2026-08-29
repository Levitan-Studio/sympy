from sympy.core.numbers import Rational
from sympy.core.symbol import Symbol
from sympy.functions.elementary.trigonometric import sin
from sympy.matrices.dense import Matrix
from sympy.matrices.exceptions import ShapeError
from sympy.matrices.expressions.determinant import Determinant
from sympy.matrices.expressions.matadd import MatAdd
from sympy.matrices.expressions.matmul import MatMul
from sympy.matrices.expressions.matpow import MatPow
from sympy.matrices.expressions.transpose import Transpose
from sympy.matrices.immutable import ImmutableDenseMatrix
from sympy.matrices.matrices_uneval import UnevaluatedMatrix
from sympy.printing.latex import latex
from sympy.printing.str import sstr
from sympy.testing.pytest import raises

x = Symbol('x')

A = UnevaluatedMatrix([[1, 2], [3, 4]])
B = UnevaluatedMatrix([[1, 5], [2, 1]])


def test_construction():
    assert A.args == (ImmutableDenseMatrix([[1, 2], [3, 4]]),)
    assert A.matrix == ImmutableDenseMatrix([[1, 2], [3, 4]])
    assert A == UnevaluatedMatrix(Matrix([[1, 2], [3, 4]]))
    assert A == UnevaluatedMatrix(2, 2, [1, 2, 3, 4])
    assert hash(A) == hash(UnevaluatedMatrix([[1, 2], [3, 4]]))
    assert A != B


def test_matrix_interface():
    assert A.shape == (2, 2)
    assert (A.rows, A.cols) == (2, 2)
    assert A.is_Matrix and A.is_MatrixExpr
    assert A.is_square
    assert A[0, 1] == 2
    assert A[3] == 4
    assert A.as_explicit() == ImmutableDenseMatrix([[1, 2], [3, 4]])
    assert UnevaluatedMatrix([[1, 2, 3]]).shape == (1, 3)


def test_add_stays_unevaluated():
    assert isinstance(A + B, MatAdd)
    assert (A + B).args == (A, B)
    assert (A + B).doit() == ImmutableDenseMatrix([[2, 7], [5, 5]])
    assert (A - B).doit() == ImmutableDenseMatrix([[0, -3], [1, 3]])
    assert (-A).doit() == ImmutableDenseMatrix([[-1, -2], [-3, -4]])
    raises(ShapeError, lambda: A + UnevaluatedMatrix([[1, 2, 3]]))


def test_mixed_with_explicit_matrix():
    # UnevaluatedMatrix wins over Matrix on both sides
    assert isinstance(A + Matrix([[1, 5], [2, 1]]), MatAdd)
    assert isinstance(Matrix([[1, 5], [2, 1]]) + A, MatAdd)
    assert (A + Matrix([[1, 5], [2, 1]])).doit() == \
        ImmutableDenseMatrix([[2, 7], [5, 5]])


def test_mul_stays_unevaluated():
    assert isinstance(A*B, MatMul)
    assert (A*B).doit() == ImmutableDenseMatrix([[5, 7], [11, 19]])
    assert (B*A).doit() == ImmutableDenseMatrix([[16, 22], [5, 8]])
    assert (A @ B).doit() == (A*B).doit()
    assert isinstance(2*A, MatMul)
    assert (2*A).doit() == ImmutableDenseMatrix([[2, 4], [6, 8]])
    assert (A/2).doit() == ImmutableDenseMatrix(
        [[Rational(1, 2), 1], [Rational(3, 2), 2]])


def test_pow_stays_unevaluated():
    assert isinstance(A**2, MatPow)
    assert (A**2).doit() == ImmutableDenseMatrix([[7, 10], [15, 22]])
    assert (A**-1).doit() == A.matrix.inv()


def test_transpose_det_inverse():
    assert isinstance(A.T, Transpose)
    assert A.T.doit() == ImmutableDenseMatrix([[1, 3], [2, 4]])
    assert isinstance(A.det(), Determinant)
    assert A.det().doit() == -2
    assert A.inv().doit() == A.matrix.inv()


def test_doit_is_deep():
    C = UnevaluatedMatrix([[A.det(), 0], [0, 1]])
    assert C.doit() == ImmutableDenseMatrix([[-2, 0], [0, 1]])
    assert (A + B + A).doit() == ImmutableDenseMatrix([[3, 9], [8, 9]])


def test_subs_and_diff():
    C = UnevaluatedMatrix([[x, 1], [0, sin(x)]])
    assert C.subs(x, 0) == UnevaluatedMatrix([[0, 1], [0, 0]])
    assert C.free_symbols == {x}
    assert C.diff(x) == UnevaluatedMatrix([[1, 0], [0, sin(x).diff(x)]])
    assert (C + C).subs(x, 0).doit() == ImmutableDenseMatrix([[0, 2], [0, 0]])


def test_evalf():
    C = UnevaluatedMatrix([[Rational(1, 2)]])
    assert C.evalf() == UnevaluatedMatrix([[0.5]])


def test_printing():
    assert sstr(A) == sstr(A.matrix)
    assert latex(A) == latex(A.matrix)
    assert latex(A + B) == latex(A.matrix) + ' + ' + latex(B.matrix)
