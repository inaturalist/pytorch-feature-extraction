#!/usr/bin/env python

from feature_extraction import extract_prelogits_fv
from scipy.spatial.distance import euclidean
import h5py 

my_prelogits = extract_prelogits_fv("mine.jpg")
f = h5py.File('features.h5', 'r')

nearest_idx = 0
nearest_distance = euclidean(f['PreLogits'][0], my_prelogits)
furthest_idx = 0
furthest_distance = nearest_distance

for i in range(1, len(f['PreLogits'])):
    distance = euclidean(f['PreLogits'][i], my_prelogits)
    if distance < nearest_distance:
        nearest_distance = distance
        nearest_idx = i
    if distance > furthest_distance:
        furthest_distance = distance
        furthest_idx = i

print("nearest distance: {}".format(nearest_distance))
print("nearest file: {}".format(f['filenames'][nearest_idx]))
print("furthest distance: {}".format(furthest_distance))
print("furthest file: {}".format(f['filenames'][furthest_idx]))
