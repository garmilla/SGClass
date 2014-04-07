import numpy as np
from HBsep import HBsep
import pdb
from datetime import datetime
import pickle
import cProfile

filter_list_path = 'filter_list_path'
filter_list_names = 'filter_list_names'
list_of_sed_list_paths = ['sed/star', 'sed/galaxy', 'sed/qso']
data = 'result_patch.txt'
z_median = [0.0, 0.3, 1.0]
z_pow = [0.0, 2.5, 2.5]

#pdb.set_trace()
classifier = HBsep()
classifier.get_filter_norm(filter_list_path, filter_list_names)
classifier.create_models(filter_list_path, list_of_sed_list_paths)
classifier.read_and_process_data(data)
classifier.fit_models()
#classifier.coefficient_marginalization()
#cProfile.run("func, xopt = classifier.optimize(z_median, z_pow)")
func, xopt = classifier.optimize(z_median, z_pow)
dt = datetime.now()
fname = 'classifications/{0}.fit'.format(dt.strftime('%y%m%d%H%M'))
classifier.write_relative_likelihoods(fname)
fname = 'classifications/{0}.pkl'.format(dt.strftime('%y%m%d%H%M'))
with open(fname, 'w') as f:
    pickle.dump(classifier, f)
