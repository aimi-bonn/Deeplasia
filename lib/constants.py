""""
script to declare constants (e.g. paths or labels)
"""
from os import path
from os.path import abspath, dirname

# --------- Base Directories ---------

# paths to project base dir (= cd ../../.. -> "~/bone2gene/")
path_to_base_dir = dirname(dirname(dirname(abspath(__file__))))

path_to_data_dir = path.join(path_to_base_dir, "data")

path_to_log_dir = path.join(path_to_base_dir, "bone_age", "logs")

path_to_data_management_dir = path.join(path_to_base_dir, "bone_age", "data-management")

# --------- Utils ---------
path_to_rsna_dir = path.join(path_to_data_dir, "annotated", "rsna_bone_age")

# path to "data/tmp/pickle_obj/" directory
path_to_pickle_dir = path.join(path_to_data_dir, "tmp", "pickle_obj")

# path to "data/tmp/csv/" directory
path_to_csv_dir = path.join(path_to_data_dir, "tmp", "csv")
