from abcraster.base import run
import subprocess
from pytest import approx


def test_run_workflow(reference_path, flood_path, aoi_path, out_dir):
    df = run(input_data_filepaths=[flood_path], ref_data_filepath=reference_path, out_dirpath=out_dir,
             metrics_list=['OA', 'K', 'CSI', 'UA', 'PA'], aoi_filepath=aoi_path)

    assert (df.loc['Overall Accuracy'][0] == approx(0.9704, abs=0.0001))
    assert (df.loc['Kappa'][0] == approx(0.8135, abs=0.0001))
    assert (df.loc['Critical Success Index'][0] == approx(0.7090, abs=0.0001))
    assert (df.loc['Users Accuracy'][0] == approx(0.8109, abs=0.0001))
    assert (df.loc['Producers Accuracy'][0] == approx(0.8494, abs=0.0001))


def test_command_line_interface(reference_path, flood_path, aoi_path, out_dir):
    out_csv_file = out_dir / 'out_mult.csv'
    out_conf_file = out_dir / 'out_conf.tif'
    comm = 'abcraster -in {} -ref {} -aoi {} -csv {} --metrics {} -out'
    comm = comm.format(flood_path.as_posix(), reference_path.as_posix(), aoi_path.as_posix(),
                       out_csv_file.as_posix(), ', '.join(['OA', 'K', 'CSI', 'UA', 'PA']), out_conf_file)

    result = subprocess.run(comm.split())

    if result.returncode == 1:
        raise RuntimeError
