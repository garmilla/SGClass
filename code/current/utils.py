import numpy as np
import ctypes as ct

def ctype_2D_double_pointer(arr):
    assert isinstance(arr,np.ndarray)
    ctarr = arr.shape[0]*[None]
    for i in range(arr.shape[0]):
        ctarr[i] = np.ctypeslib.as_ctypes(arr[i,:])
    arrp = (ct.POINTER(ct.c_double)*arr.shape[0])(*ctarr)
    return arrp

def get_train_and_test(Ndata,truth,label_types,fraction,
                       rel_fractions=None):

    # shuffle
    ind = np.random.permutation(Ndata)

    # get training indicies
    Ntrain = np.int(np.round(Ndata * fraction))
    Ntest  = Ndata - Ntrain
    train_ind = ind[:Ntrain]
    train_labels = truth[train_ind]

    # organize training data into classes
    class_train_ind = {}
    for l in label_types:
        class_train_ind[l] = np.sort(train_ind[
                np.where(train_labels==l)[0]])
    
    # relative fractions?
    if rel_fractions!=None:
        assert isinstance(rel_fractions,np.ndarray)
        N = np.array([class_train_ind[l].shape[0] for l in label_types])
        idx = np.where(N==N.min())[0]
        rel_fractions /= rel_fractions[idx]
        N = np.round(rel_fractions * N.min())
        train_ind = np.array([],dtype=np.int)
        for i,l in enumerate(label_types):
            class_train_ind[l] = class_train_ind[l][:N[i]]
            train_ind = np.append(train_ind,class_train_ind[l])

    # sort and get test indicies
    test_ind = np.sort(ind)
    train_ind = np.sort(train_ind)
    test_ind = np.delete(test_ind,train_ind)
    test_labels = truth[test_ind]

    # organize test data into classes
    class_test_ind = {}
    for l in label_types:
        class_test_ind[l] = np.sort(test_ind[
                np.where(test_labels==l)[0]])

    return class_train_ind,class_test_ind,train_ind,test_ind

def quick_ML_assess(star_chi2,gal_chi2,truth):

    correct = 0.
    cstar = 0.
    cgalaxy = 0.
    for i in range(truth.shape[0]):
        if ((star_chi2[i].min()<gal_chi2[i].min()) & (truth[i]==1)):
            correct += 1.
            cstar += 1.
        if ((star_chi2[i].min()>gal_chi2[i].min()) & (truth[i]==0)):
            correct += 1.
            cgalaxy += 1.
    ind = np.where(truth!=-99)
    print 'ML total'
    print correct, truth[ind].shape[0],correct/truth[ind].shape[0]
    ind = np.where(truth==1)[0]
    print 'ML star'
    print cstar,truth[ind].shape[0],cstar/truth[ind].shape[0]
    ind = np.where(truth==0)[0]
    print 'ML gal'
    print cgalaxy,truth[ind].shape[0],cgalaxy/truth[ind].shape[0],'\n'

def quick_HB_assess(star_prob,gal_prob,truth):
    correct = 0.
    cstar = 0.
    cgalaxy = 0.
    for i in range(truth.shape[0]):
        if ((star_prob[i]>gal_prob[i]) & (truth[i]==1)):
            correct += 1.
            cstar+=1.
        if ((star_prob[i]<gal_prob[i]) & (truth[i]==0)):
            correct += 1.
            cgalaxy += 1.
    ind = np.where(truth!=-99)
    print 'HB total'
    print correct, truth[ind].shape[0],correct/truth[ind].shape[0]
    ind = np.where(truth==1)[0]
    print 'HB star'
    print cstar,truth[ind].shape[0],cstar/truth[ind].shape[0]
    ind = np.where(truth==0)[0]
    print 'HB gal'
    print cgalaxy,truth[ind].shape[0],cgalaxy/truth[ind].shape[0],'\n'

def eq2gal(ra,dec):
    """
    Convert Equatorial coordinates to Galactic Coordinates in the epch J2000.
    
    Keywords arguments:
    ra  -- Right Ascension (in radians)
    dec -- Declination (in radians)

    Return a tuple (l, b):
    l -- Galactic longitude (in radians)
    b -- Galactic latitude (in radians)
    """
   # RA(radians),Dec(radians) of Galactic Northpole in J2000
    Galactic_Northpole_Equatorial=(np.radians(192.859508), np.radians(27.128336))

    alpha = Galactic_Northpole_Equatorial[0]
    delta = Galactic_Northpole_Equatorial[1]
    la = np.radians(33.0)
    
    b = np.arcsin(np.sin(dec) * np.sin(delta) +
                  np.cos(dec) * np.cos(delta) * np.cos(ra - alpha))

    l = np.arctan2(np.sin(dec) * np.cos(delta) - 
                   np.cos(dec) * np.sin(delta) * np.cos(ra - alpha), 
                   np.cos(dec) * np.sin(ra - alpha)
                   ) + la

    l = l if l >= 0 else (l + np.pi * 2.0)

    l = l % (2.0 * np.pi)

    return l, b

def csv_to_txt(Ifname, output_bands):
    Ofname = Ifname[:-3]
    Ofname += 'txt'
    
    bands = []
    bands_ind = []
    output_bands_ind = []
    bands_err_ind = []
    output_bands_err_ind = []
    with open(Ifname, 'r') as If:
        reader = csv.reader(If)
        line = reader.next()
        if 'u' in line:
    	    bands.append('u')
            bands_ind.append(line.index('u'))
            bands_err_ind.append(line.index('err_u'))
        if 'g' in line:
    	    bands.append('g')
            bands_ind.append(line.index('g'))
            bands_err_ind.append(line.index('err_g'))
        if 'r' in line:
    	    bands.append('r')
            bands_ind.append(line.index('r'))
            bands_err_ind.append(line.index('err_r'))
        if 'i' in line:
    	    bands.append('i')
            bands_ind.append(line.index('i'))
            bands_err_ind.append(line.index('err_i'))
        if 'z' in line:
       	    bands.append('z')
            bands_ind.append(line.index('z'))
            bands_err_ind.append(line.index('err_z'))
        Nfilters = len(bands_ind)
        for band in output_bands:
            if band not in bands:
    	        print "Error: The output band {0} was not found in the data".format(band)
    	        sys.exit(1)
    	output_bands_ind.append(bands_ind[bands.index(band)])
    	output_bands_err_ind.append(bands_err_ind[bands.index(band)])
        print "Data for bands:", bands
        print "Output data for bands:", output_bands
        with open(Ofname, 'w') as Of:
            for line in reader:
    	        data_line = ''
    	    for i in output_bands_ind:
    	        data_line += line[i] + ' '
    	    for i in output_bands_err_ind:
    	        data_line += line[i] + ' '
    	    data_line = data_line[:-1]
    	    data_line += '\n'
    	    Of.write(data_line)

def StarPrior(Ifname, filters_file='filter_list_names'):
    """
    Create a function that approximates the distribution of stars in flux space in the bands provided
    """
    f = open(filters_file)
    filter_names = f.readlines()

    bands = []
    bands_ind = []
    output_bands_ind = []
    bands_err_ind = []
    output_bands_err_ind = []
    with open(Ifname, 'r') as If:
        reader = csv.reader(If)
        line = reader.next()
        if 'u' in line:
    	    bands.append('u')
            bands_ind.append(line.index('u'))
            bands_err_ind.append(line.index('err_u'))
        if 'g' in line:
    	    bands.append('g')
            bands_ind.append(line.index('g'))
            bands_err_ind.append(line.index('err_g'))
        if 'r' in line:
    	    bands.append('r')
            bands_ind.append(line.index('r'))
            bands_err_ind.append(line.index('err_r'))
        if 'i' in line:
    	    bands.append('i')
            bands_ind.append(line.index('i'))
            bands_err_ind.append(line.index('err_i'))
        if 'z' in line:
       	    bands.append('z')
            bands_ind.append(line.index('z'))
            bands_err_ind.append(line.index('err_z'))
        Nfilters = len(bands_ind)
        for band in output_bands:
            if band not in bands:
    	        print "Error: The output band {0} was not found in the data".format(band)
    	        sys.exit(1)
    	output_bands_ind.append(bands_ind[bands.index(band)])
    	output_bands_err_ind.append(bands_err_ind[bands.index(band)])
        print "Data for bands:", bands
        print "Output data for bands:", output_bands
        with open(Ofname, 'w') as Of:
            for line in reader:
    	        data_line = ''
    	    for i in output_bands_ind:
    	        data_line += line[i] + ' '
    	    for i in output_bands_err_ind:
    	        data_line += line[i] + ' '
    	    data_line = data_line[:-1]
    	    data_line += '\n'
    	    Of.write(data_line)
