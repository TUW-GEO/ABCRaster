import pytest
from pytest import approx
from abcraster.metrics import *
import os
from abcraster.base import run
import subprocess

#requires pytes and pytest-cov to be installed

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

def test_raster_shapfile():
    """
    test data from greece paper flood mapping result
    validation data retrieved from Copernius emergency Service:
     https://emergency.copernicus.eu/mapping/list-of-components/EMSR271
    """
    data_path = os.path.join(os.path.dirname(__file__), 'data')

    shp = 'EMSR271_02FARKADONA_DEL_v1_observed_event_a.shp'
    tif = 'FLOOD-HM_20180228T163112__VV_A175_E054N006T3_EU020M_V0M0R1_S1.tif'
    aoi = 'EMSR271_02FARKADONA_DEL_v1_area_of_interest_a.shp'

    shp_path = os.path.join(data_path, shp)
    tif_path = os.path.join(data_path, tif)
    aoi_path = os.path.join(data_path, aoi)

    temp_path = os.path.join(data_path, 'tmp')
    if not os.path.exists(temp_path):
        os.makedirs(temp_path)

    df = run([tif_path], shp_path, temp_path, metrics.keys(),
             aoi_filepath=aoi_path, delete_tmp_files=True)

    assert (df.loc['Overall Accuracy'][0] == approx(0.9704, abs=0.0001))
    assert (df.loc['Kappa'][0] == approx(0.8135, abs=0.0001))
    assert (df.loc['Critical Success Index'][0] == approx(0.7090, abs=0.0001))
    assert (df.loc['Bias'][0] == approx(1.0474, abs=0.0001))
    assert (df.loc['Prevalence'][0] == approx(0.0848, abs=0.0001))
    assert (df.loc['Users Accuracy'][0] == approx(0.8109, abs=0.0001))
    assert (df.loc['Producers Accuracy'][0] == approx(0.8494, abs=0.0001))
    assert (df.loc['commission error'][0] == approx(0.1891, abs=0.0001))
    assert (df.loc['omission error'][0] == approx(0.1506, abs=0.0001))
    assert (df.loc['Success Rate'][0] == approx(0.7211, abs=0.0001))
    assert (df.loc['F1 Score'][0] == approx(0.8297, abs=0.0001))
    assert (df.loc['True Negative Rate'][0] == approx(0.9816, abs=0.0001))
    assert (df.loc['False Positive Rate'][0] == approx(0.0184, abs=0.0001))
    assert (df.loc['Negative Predictive Value'][0] == approx(0.9860, abs=0.0001))
    assert (df.loc['False Omission Rate'][0] == approx(0.0140, abs=0.0001))


def test_command_line():
    """ test commandline interface """

    data_path = os.path.join(os.path.dirname(__file__), 'data')

    shp = 'EMSR271_02FARKADONA_DEL_v1_observed_event_a.shp'
    tif1 = 'FLOOD-HM_20180228T163112__VV_A175_E054N006T3_EU020M_V0M0R1_S1.tif'
    tif2 = 'FLOOD-HM_20180228T163112__VV_A175_E054N006T3_EU020M_V0M0R1_S1_Copy.tif'
    aoi = 'EMSR271_02FARKADONA_DEL_v1_area_of_interest_a.shp'

    shp_path = os.path.join(data_path, shp)
    tif_path1 = os.path.join(data_path, tif1)
    tif_path2 = os.path.join(data_path, tif2)
    tif_paths = '{} {}'.format(tif_path1, tif_path2)
    aoi_path = os.path.join(data_path, aoi)

    temp_path = os.path.join(data_path, 'tmp')
    if not os.path.exists(temp_path):
        os.makedirs(temp_path)

    diff_path = os.path.join(temp_path,'val.tif')

    comm = 'abcraster -in {} -ref {} -aoi {} -out {} -csv {}'.format(tif_paths, shp_path,
                                                                                    aoi_path, diff_path,
                                                                                           'out_mult.csv')

    result = subprocess.run(comm.split())

    if result.returncode == 1:
        raise RuntimeError