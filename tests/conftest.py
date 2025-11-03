import pytest
from pathlib import Path


@pytest.fixture
def approval_input_directory():
    return Path(r'/eodc/private/tuwgeo/software/ci/abcraster/input')


@pytest.fixture
def out_dir():
    out_path = Path(r'/tmp/abcraster')
    out_path.mkdir(exist_ok=True)
    return out_path


@pytest.fixture
def reference_path(approval_input_directory):
    return approval_input_directory / 'EMSR271_02FARKADONA_DEL_v1_observed_event_a.shp'


@pytest.fixture
def flood_path(approval_input_directory):
    return approval_input_directory / 'FLOOD-HM_20180228T163112__VV_A175_E054N006T3_EU020M_V0M0R1_S1.tif'


@pytest.fixture
def aoi_path(approval_input_directory):
    return approval_input_directory / 'EMSR271_02FARKADONA_DEL_v1_area_of_interest_a.shp'
