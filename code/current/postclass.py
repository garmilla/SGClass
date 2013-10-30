import matplotlib.pyplot as plt
import pyfits
import numpy as np
import sys
import csv
import pdb

class PhotoObj:
# Class that contains the list of objects in a classification run
    def __init__(self, classifications=None, catalogue=None, sdss=True):
    # Initialize the class by loading the data in the classifications file and
    # the catalogue file.
        if classifications == None:
	    print "Type absolute path to the classifications file (fits file)."
	    classifications = raw_input('classifications-->')
        if catalogue == None:
	    print "Type absolute path to the corresponding catalogue file (csv file)."
	    catalogue = raw_input('catalogue-->')
	try:
            with pyfits.open(classifications) as f:
                self.tdata = f[1].data
        except:
	    print "Failed to open and load file {0}".format(classifications)
	    self.__init__(classifications, catalogue)
	try:
            with open(catalogue, 'rb') as f:
	        reader = csv.reader(f)
		self.cat_columns = reader.next()
		self.cdata = []
		i = 0
		for row in reader:
		    self.cdata.append(row)
		    i += 1
        except:
	    print "Failed to open and load file {0}".format(catalogue)
	    self.__init__(classifications, catalogue)
	assert i==self.tdata.shape[0],\
	'The number of objects in the {0} and {1} do not match'.format(classifications, catalogue)
	self.Nobj = i
	self.build_list()

    def build_list(self):
    # Build list that contains the objects
        self.Objs = []
	for i in range(self.Nobj):
	# Store each object in a dictionary
	    Obj = {}
	    Obj['class'] = []
	    for value in self.tdata[i]:
	        Obj['class'].append(value)
	    if Obj['class'][3] != 0.0:
	        indx = Obj['class'][0:3].index(max(Obj['class'][0:3]))
		if indx == 0:
	            Obj['ctype'] = 'star'
		elif indx == 1:
	            Obj['ctype'] = 'gal'
		elif indx == 2:
	            Obj['ctype'] = 'qso'
		else:
	            Obj['ctype'] = 'unkwn'
	    else:
	        Obj['ctype'] = 'unkwn'
	    Obj['class'] = self.tdata[i]
	    for (j, value) in enumerate(self.cdata[i]):
	        Obj[self.cat_columns[j]] = value
	    self.Objs.append(Obj)

    def __iter__(self):
    # Make the class itself iterable
        Iter = iter(self.Objs)
        return Iter

    def select_type(self, type='qso'):
        ind = []
	for (i, obj) in enumerate(self):
	    if obj['ctype'] == type:
	        ind.append(i)
        return ind

    def select_colors(self, colors=['g','r','r','i'],ranges=[[1.2,1.6],[0.8,1.3]]):
        ind = []
	for (i, obj) in enumerate(self):
	    color1 = float(obj[colors[0]]) - float(obj[colors[1]])
	    color2 = float(obj[colors[2]]) - float(obj[colors[3]])
	    if color1 > ranges[0][0] and\
	       color1 < ranges[0][1] and\
	       color2 > ranges[1][0] and\
	       color2 < ranges[1][1]:
	        ind.append(i)
        return ind

    def select_typeclrranges(self, type=None, colors=None, ranges=None):
        if type == None:
	    type_ind = self.select_type()
	else:
	    type_ind = self.select_type(type)
	if colors == None or ranges == None:
	    colors_ind = self.select_colors()
	else:
	    colors_ind = self.select_colors(colors, ranges)
	ind = set(type_ind).intersection(set(colors_ind))
	return ind
