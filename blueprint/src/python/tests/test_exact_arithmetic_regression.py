"""Rational bounds and interior samples must remain exact."""
from fractions import Fraction as F
import sympy
from functions import Affine, Interval, RationalFunction as R


def run_regressions():
    for numerator in range(-5, 6):
        for denominator in range(1, 9):
            value = R([numerator], [denominator]).at(0)
            assert value == F(numerator, denominator), (numerator, denominator, value)
    third = R([1], [3])
    assert third.add(R([1], [6])).at(4) == F(1, 2)
    assert third.mul(R([3], [7])).at(4) == F(1, 7)
    assert third.div(2).at(4) == F(1, 6)

    # Floating division rounds these distinct constants to the same number,
    # incorrectly retaining the first bound for the complete envelope.
    above_one = R([2**53 + 1], [2**53])
    one = R([1])
    domain = Interval(0, 1, True, True)
    for method, inputs, expected in (
        (R.min, [above_one, one], 1), (R.max, [one, above_one], 1),
    ):
        pieces = method([(f, domain) for f in inputs], domain)
        for point in (0, F(1, 3), F(1, 2), 1):
            active = [(f, ref) for f, cell, ref in pieces if cell.contains(point)]
            assert len(active) == 1
            assert active[0][1] == expected

    # A rounded midpoint can land on the excluded endpoint of a nonempty
    # interval, causing the affine envelope to drop the entire function.
    for base in (2**60, -(2**60)):
        interval = Interval(base, base + 2, False, False)
        assert interval.contains(interval.midpoint())
        assert interval.midpoint() == base + 1
        affine = Affine(0, 7, interval)
        for method in (affine.min_with, affine.max_with):
            pieces = method([], interval)
            assert len(pieces) == 1 and pieces[0].domain == interval
            assert pieces[0].at(base + 1) == 7
    assert Interval(F(1, 3), F(2, 3)).midpoint() == F(1, 2)
    assert Interval(sympy.sqrt(2), sympy.sqrt(3)).midpoint() == (sympy.sqrt(2) + sympy.sqrt(3)) / 2
    assert Interval(0.25, 0.75).midpoint() == 0.5


run_regressions()
