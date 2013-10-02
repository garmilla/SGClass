import numpy as np
from HBsep import HBsep
import pdb


Nzs = [1, 10]
z_maxs = [0.0, 7.0]
class_labels = ['stars', 'galaxy_and_qsos']
filter_list_path = 'filter_list_path'
list_of_sed_list_paths = ['sed_list_star_path', 'sed_list_galaxy_path']
data = 'result.txt'
method = 1
filename = 'output.fit'
z_median = [0.0, 3.5]
z_pow = [0.0, 0.0]

pdb.set_trace()
classifier = HBsep(class_labels, Nzs, z_maxs)
classifier.get_filter_norm(filter_list_path)
classifier.create_models(filter_list_path, list_of_sed_list_paths)
classifier.read_and_process_data(data)
classifier.fit_models()
classifier.coefficient_marginalization()
classifier.optimize(z_median, z_pow)
classifier.write_relative_likelihoods(filename)
