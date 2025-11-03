from abcraster.base import run, Validation
from abcraster.metrics import metrics
import subprocess
from pytest import approx


def test_class_workflow(approval_input_directory, reference_path, flood_path, aoi_path, out_dir):
    val = Validation(input_data_filepath=flood_path, ref_data_filepath=reference_path,
                     out_dirpath=out_dir)
    val.accuracy_assessment()
    oa_raw = val.calculate_accuracy_metric(metrics['OA'])
    assert (oa_raw == approx(0.9915, abs=0.0001))

    mask_path = approval_input_directory / 'handmask_clipped.tif'
    val.apply_mask(mask_path)
    val.apply_mask(aoi_path, invert_mask=True)
    val.accuracy_assessment()
    oa_masked = val.calculate_accuracy_metric(metrics['OA'])
    assert (oa_masked == approx(0.9669, abs=0.0001))

    sampling_path = approval_input_directory / 'sampling_2500_2500.tif'
    val.load_sampling(sampling_path)
    val.accuracy_assessment()
    oa_sampled = val.calculate_accuracy_metric(metrics['OA'])
    assert (oa_sampled == approx(0.9114, abs=0.0001))


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
    comm = 'abcraster -in {} -ref {} -aoi {} -csv {} --metrics {} -out {}'
    comm = comm.format(flood_path.as_posix(), reference_path.as_posix(), aoi_path.as_posix(),
                       out_csv_file.as_posix(), ', '.join(['OA', 'K', 'CSI', 'UA', 'PA']), out_conf_file.as_posix())

    result = subprocess.run(comm.split())

    if result.returncode == 1:
        raise RuntimeError
