#!/usr/bin/env python3

import os
import json
import numpy as np
import nibabel as nib
from matplotlib import cm
from scipy.io import loadmat
from dipy.segment.clustering import QuickBundles
from dipy.tracking.streamline import length
from dipy.tracking.metrics import mean_curvature, mean_orientation
from dipy.io.streamline import load_tractogram
from dipy.tracking.utils import density_map
import pandas as pd

# compute stats and output csv file
def computeStats(subjectID,reference,streamlines,classification,outdir):

	avg_length = []
	avg_curv = []
	stream_count = []
	mean_x = []
	mean_y = []
	mean_z = []
	num_vox = []

	tract_index_labels = np.trim_zeros(np.unique(classification['index'].tolist()))

	qb = QuickBundles(np.inf)

	for i in tract_index_labels:
		indices = [ t for t in range(len(classification['index'].tolist())) if classification['index'].tolist()[int(t)] == i ]
		avg_length.append(np.mean(length(streamlines[indices])))
		
		clusters = qb.cluster(streamlines[indices])
		avg_curv.append(mean_curvature(clusters.centroids[0]))
		orientation = mean_orientation(clusters.centroids[0])
		mean_x.append(orientation[0])
		mean_y.append(orientation[1])
		mean_z.append(orientation[2])
		stream_count.append(len(indices))
		denmap = density_map(streamlines[indices],reference.affine,reference.shape)
		num_vox.append(len(denmap[denmap>0]))

	df = pd.DataFrame([],dtype=object)
	df['subjectID'] = [ subjectID for f in range(len(classification['names'].tolist())) ]
	df['structureID'] = [ f for f in classification['names'].tolist() ]
	df['nodeID'] = [ 1 for f in range(len(df['structureID'])) ]
	df['streamline_count'] = stream_count
	df['average_length'] = avg_length
	df['average_curvature'] = avg_curv
	df['voxel_count'] = num_vox
	df['centroid_x'] = mean_x
	df['centroid_y'] = mean_y
	df['centroid_z'] = mean_z

	df.to_csv('%s/output_FiberStats.csv' %outdir,index=False)
	

def main():
	
	# load config.json structure
	with open('config.json','r') as config_f:
		config = json.load(config_f)

	# parse inputs
	subjectID = config['_inputs'][0]['meta']['subject']
	outdir = './tractmeasures/'
	# load data
	reference = nib.load(config['dwi'])
	sft = load_tractogram(config['tractogram'],reference)
	classification = loadmat(config['classification'],squeeze_me=True)['classification']

	# make output directory if not made already
	if os.path.isdir(outdir):
		print("directory exits")
	else:
		print("making output directory")
		os.mkdir(outdir)

	# compute stats
	computeStats(subjectID,reference,sft.streamlines,classification,outdir)

if __name__ == '__main__':
	main()