import pytest
from pytest import approx
from abcraster.metrics import *


def test_overall_accuracy():
    assert overall_accuracy(dict(TP=100, TN=80, FP=10, FN=30)) == approx(0.8182, abs=0.001)
    assert overall_accuracy(dict(TP=100, TN=80, FP=0, FN=0)) == approx(1.0, abs=0.001)
    assert overall_accuracy(dict(TP=0, TN=0, FP=10, FN=10)) == approx(0.0, abs=0.001)

    with pytest.raises(ZeroDivisionError):
        overall_accuracy(dict(TP=0, TN=0, FP=0, FN=0))


def test_kappa():
    assert kappa(dict(TP=100, TN=80, FP=10, FN=30)) == approx(0.6364, abs=0.001)
    assert kappa(dict(TP=100, TN=80, FP=0, FN=0)) == approx(1.0, abs=0.001)
    assert kappa(dict(TP=0, TN=0, FP=10, FN=10)) == approx(-1.0, abs=0.001)

    with pytest.raises(ZeroDivisionError):
        kappa(dict(TP=0, TN=0, FP=0, FN=0))


def test_users_accuracy():
    assert users_accuracy(dict(TP=100, TN=80, FP=10, FN=30)) == approx(0.9090, abs=0.001)
    assert users_accuracy(dict(TP=100, TN=80, FP=0, FN=0)) == approx(1.0, abs=0.001)
    assert users_accuracy(dict(TP=0, TN=0, FP=10, FN=10)) == approx(0.0, abs=0.001)

    with pytest.raises(ZeroDivisionError):
        users_accuracy(dict(TP=0, TN=0, FP=0, FN=0))


def test_producers_accuracy():
    assert producers_accuracy(dict(TP=100, TN=80, FP=10, FN=30)) == approx(0.7692, abs=0.001)
    assert producers_accuracy(dict(TP=100, TN=80, FP=0, FN=0)) == approx(1.0, abs=0.001)
    assert producers_accuracy(dict(TP=0, TN=0, FP=10, FN=10)) == approx(0.0, abs=0.001)

    with pytest.raises(ZeroDivisionError):
        producers_accuracy(dict(TP=0, TN=0, FP=0, FN=0))


def test_critical_success_index():
    assert critical_success_index(dict(TP=100, TN=80, FP=10, FN=30)) == approx(0.7143, abs=0.001)
    assert critical_success_index(dict(TP=100, TN=80, FP=0, FN=0)) == approx(1.0, abs=0.001)
    assert critical_success_index(dict(TP=0, TN=0, FP=10, FN=10)) == approx(0.0, abs=0.001)

    with pytest.raises(ZeroDivisionError):
        critical_success_index(dict(TP=0, TN=0, FP=0, FN=0))


def test_f1_score():
    assert f1_score(dict(TP=100, TN=80, FP=10, FN=30)) == approx(0.8333, abs=0.001)
    assert f1_score(dict(TP=100, TN=80, FP=0, FN=0)) == approx(1.0, abs=0.001)
    assert f1_score(dict(TP=0, TN=0, FP=10, FN=10)) == approx(0.0, abs=0.001)

    with pytest.raises(ZeroDivisionError):
        producers_accuracy(dict(TP=0, TN=0, FP=0, FN=0))


def test_commission_error():
    assert commission_error(dict(TP=100, TN=80, FP=10, FN=30)) == approx(0.0909, abs=0.001)
    assert commission_error(dict(TP=100, TN=80, FP=0, FN=0)) == approx(0.0, abs=0.001)
    assert commission_error(dict(TP=0, TN=0, FP=10, FN=10)) == approx(1.0, abs=0.001)

    with pytest.raises(ZeroDivisionError):
        commission_error(dict(TP=0, TN=0, FP=0, FN=0))


def test_omission_error():
    assert omission_error(dict(TP=100, TN=80, FP=10, FN=30)) == approx(0.2308, abs=0.001)
    assert omission_error(dict(TP=100, TN=80, FP=0, FN=0)) == approx(0.0, abs=0.001)
    assert omission_error(dict(TP=0, TN=0, FP=10, FN=10)) == approx(1.0, abs=0.001)

    with pytest.raises(ZeroDivisionError):
        omission_error(dict(TP=0, TN=0, FP=0, FN=0))


def test_penalization():
    assert penalization(dict(TP=100, TN=80, FP=10, FN=30)) == approx(0.9481, abs=0.001)
    assert penalization(dict(TP=100, TN=80, FP=0, FN=0)) == approx(1.0, abs=0.001)
    assert penalization(dict(TP=0, TN=0, FP=10, FN=10)) == approx(0.5, abs=0.001)

    with pytest.raises(ZeroDivisionError):
        penalization(dict(TP=0, TN=0, FP=0, FN=0))


def test_success_rate():
    assert success_rate(dict(TP=100, TN=80, FP=10, FN=30)) == approx(0.7173, abs=0.001)
    assert success_rate(dict(TP=100, TN=80, FP=0, FN=0)) == approx(1.0, abs=0.001)
    assert success_rate(dict(TP=0, TN=0, FP=10, FN=10)) == approx(-0.5, abs=0.001)

    with pytest.raises(ZeroDivisionError):
        success_rate(dict(TP=0, TN=0, FP=0, FN=0))


def test_bias():
    assert bias(dict(TP=100, TN=80, FP=10, FN=30)) == approx(0.8462, abs=0.001)
    assert bias(dict(TP=100, TN=80, FP=0, FN=0)) == approx(1.0, abs=0.001)
    assert bias(dict(TP=0, TN=0, FP=10, FN=10)) == approx(1.0, abs=0.001)

    with pytest.raises(ZeroDivisionError):
        bias(dict(TP=0, TN=0, FP=0, FN=0))


def test_prevalence():
    assert prevalence(dict(TP=100, TN=80, FP=10, FN=30)) == approx(0.5909, abs=0.001)
    assert prevalence(dict(TP=100, TN=80, FP=0, FN=0)) == approx(0.5556, abs=0.001)
    assert prevalence(dict(TP=0, TN=0, FP=10, FN=10)) == approx(0.5, abs=0.001)

    with pytest.raises(ZeroDivisionError):
        prevalence(dict(TP=0, TN=0, FP=0, FN=0))
