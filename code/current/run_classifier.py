import numpy as np
from HBsep import HBsep
import pdb
from datetime import datetime
import pickle

Nzs = [1, 30, 30]
z_maxs = [0.0, 2.0, 5.0]
class_labels = ['star', 'galaxy', 'qso']
filter_list_path = 'filter_list_path'
list_of_sed_list_paths = ['sed/star', 'sed/galaxy', 'sed/qso']
data = 'result_patch.txt'
method = 1
z_median = [0.0, 0.3, 1.0]
z_pow = [0.0, 2.5, 2.5]

#pdb.set_trace()
classifier = HBsep(class_labels, Nzs, z_maxs)
classifier.get_filter_norm(filter_list_path)
classifier.create_models(filter_list_path, list_of_sed_list_paths)
classifier.read_and_process_data(data)
classifier.fit_models()
classifier.coefficient_marginalization()
func, xopt = classifier.optimize(z_median, z_pow)
dt = datetime.now()
fname = 'classifications/{0}.fit'.format(dt.strftime('%y%m%d%H%M'))
classifier.write_relative_likelihoods(fname)
fname = 'classifications/{0}.pkl'.format(dt.strftime('%y%m%d%H%M'))
with open(fname, 'w') as f:
    pickle.dump(classifier, f)
