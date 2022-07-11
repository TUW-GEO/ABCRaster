import pytest
from pytest import approx

from abcraster.metrics import overall_accuracy


def test_overall_accuracy():
    assert overall_accuracy(dict(TP=100, TN=80, FP=10, FN=30)) == approx(0.8182, abs=0.001)
    assert overall_accuracy(dict(TP=100, TN=80, FP=0, FN=0)) == approx(1.0, abs=0.001)
    assert overall_accuracy(dict(TP=0, TN=0, FP=10, FN=10)) == approx(0.0, abs=0.001)


def test_oa_all_zero():
    with pytest.raises(ZeroDivisionError):
        overall_accuracy(dict(TP=0, TN=0, FP=0, FN=0))