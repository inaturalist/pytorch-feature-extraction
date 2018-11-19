#!/usr/bin/env python

import os
import argparse
import h5py
import numpy as np

from feature_extraction import extract_prelogits_fv

parser = argparse.ArgumentParser()
parser.add_argument(
    '--images_directory', type=str,  required=True,
    help='directory of images to extract fvs for'
)
parser.add_argument(
    '--output_file', type=str, required=True,
    help='path to output file (hdf5 format)'
)

def main():
    args = parser.parse_args()
    img_dir = args.images_directory
    filenames = []
    vectors = []
    photo_ids = []
    for file in sorted(os.listdir(img_dir)):
        if "jpg" not in file:
            continue
        photo_id, ext = os.path.splitext(file)
        img_path = os.path.join(img_dir, file)
        v = extract_prelogits_fv(img_path)
        # h5py will have trouble with bare strings
        filenames.append(np.string_(img_path))
        vectors.append(v)
        photo_ids.append(int(photo_id))

    
    with h5py.File(args.output_file, 'w') as output:
        output.create_dataset("filenames", data=filenames)
        output.create_dataset("PreLogits", data=vectors)
        output.create_dataset("photo_ids", data=photo_ids)



if __name__ == "__main__":
    main()
