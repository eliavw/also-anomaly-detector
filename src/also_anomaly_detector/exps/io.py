from pathlib import Path
from affe.io import get_filename, mimic_fs, get_filepath

# Constants
ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
DATA_DIR = ROOT_DIR / "data"
QRY_DIR = ROOT_DIR / "query"
STARAI_DATA_DIR = DATA_DIR / "raw" / "datasets-starai"


def _default_data_dir():
    return DATA_DIR


def _default_query_dir():
    return QRY_DIR


# STARAI
def _default_starai_data_dir():
    return STARAI_DATA_DIR


def get_starai_dataset_names(data_dir_filepath=STARAI_DATA_DIR):
    """
    Collect all names from starai datasets
    """
    starai_datasets_dpath = data_dir_filepath / "datasets"
    starai_datasets = [d.stem for d in starai_datasets_dpath.iterdir() if d.is_dir()]
    starai_datasets.sort()
    return starai_datasets


def starai_original_filepath(
    name="nltcs", kind="train", data_dir_filepath=STARAI_DATA_DIR
):

    dataset_dpath = data_dir_filepath / "datasets" / name
    dataset_fname = get_filename([name, kind], extension="data", separator=".")
    dataset_fpath = dataset_dpath / dataset_fname

    return dataset_fpath


# OPENML_CC18

# General
def query_filepath(
    name="nltcs", keyword="default", query_dir_filepath=QRY_DIR, extension="npy"
):
    qry_fname = get_filename([name, keyword], extension=extension)
    qry_fpath = query_dir_filepath / qry_fname
    return qry_fpath


def dataset_filepath(
    name="nltcs",
    kind="train",
    step=1,
    data_dir_filepath=DATA_DIR,
    extension="csv",
    check=True,
):
    step_string = "step-{0:02d}".format(step)
    dataset_dpath = data_dir_filepath / step_string
    dataset_fname = get_filename([name, kind], extension=extension)
    dataset_fpath = dataset_dpath / dataset_fname

    if check:
        assert dataset_fpath.exists(), "Filepath: {} does not exist".format(
            dataset_fpath
        )

    return dataset_fpath
