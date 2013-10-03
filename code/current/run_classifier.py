import numpy as np
from HBsep import HBsep
import pdb

Nzs = [1, 30, 30]
z_maxs = [0.0, 1.5, 7.0]
class_labels = ['star', 'galaxy', 'qso']
filter_list_path = 'filter_list_path'
list_of_sed_list_paths = ['sed/star', 'sed/galaxy', 'sed/qso']
data = 'result.txt'
method = 1
filename = 'classification.fit'
z_median = [0.0, 1.0, 2.0]
z_pow = [0.0, 0.0, 0.0]

#pdb.set_trace()
classifier = HBsep(class_labels, Nzs, z_maxs)
classifier.get_filter_norm(filter_list_path)
classifier.create_models(filter_list_path, list_of_sed_list_paths)
classifier.read_and_process_data(data)
classifier.fit_models()
classifier.coefficient_marginalization()
classifier.optimize(z_median, z_pow)
classifier.write_relative_likelihoods(filename)
