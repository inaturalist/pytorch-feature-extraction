import datetime
import json
import magic
import os
import random
import time
import uuid
import yaml
import numpy as np
import tensorflow as tf

from flask import Flask, request, jsonify, render_template, send_from_directory
from forms import ImageForm
from PIL import Image
from feature_extraction import extract_prelogits_fv
from scipy.spatial.distance import euclidean
import h5py

config = yaml.safe_load(open("config.yml"))

app = Flask(__name__)
app.secret_key = config["app_secret"]

UPLOAD_FOLDER = "static/"

butterflies = []

class Taxon:
    taxon_names = {
        "48548": "Painted Lady",
        "48662": "Monarch",
        "49133": "Red Admiral",
        "49150": "Gulf Fritillary",
        "55626": "Cabbage White",
        "48505": "Common Buckeye",
        "60551": "Eastern Tiger Swallowtail",
        "58523": "Black Swallowtail",
        "50340": "Fiery Skipper",
        "52925": "Pearl Crescent",
    }

    def __init__(self, taxon_id):
        self.taxon_id = taxon_id
        self.tp_prelogits = []
        self.tp_photoids = []
        self.tp_filepaths = []

    def tp_filenames(self):
        return [os.path.basename(file) for file in self.tp_filepaths]

    def taxon_name(self):
        return Taxon.taxon_names[str(self.taxon_id)]

class Neighbor:
    def __init__(self, taxon, distance, photoid):
        self.taxon = taxon
        self.distance = distance
        self.photoid = photoid

butterflies = []
for bf_taxon_id in Taxon.taxon_names.keys():
    bf = Taxon(bf_taxon_id)
    h5_path = "taxon_photos/{}/features.h5".format(bf_taxon_id)
    with h5py.File(h5_path, 'r') as f:
        for i in range(len(f['PreLogits'])):
            bf.tp_prelogits.append(f['PreLogits'][i])
            bf.tp_photoids.append(f['photo_ids'][i])
            bf.tp_filepaths.append(f['filenames'][i])
    butterflies.append(bf)

@app.route('/', methods=['GET', 'POST'])
def taxon_images_nn():
    form = ImageForm()
    if request.method == 'POST':
        image_file = form.image.data
        extension = os.path.splitext(image_file.filename)[1]
        image_uuid = str(uuid.uuid4())
        file_path = os.path.join(UPLOAD_FOLDER, image_uuid) + extension
        image_file.save(file_path)

        mime_type = magic.from_file(file_path, mime=True)
        # attempt to convert non jpegs
        if mime_type != 'image/jpeg':
            im = Image.open(file_path)
            rgb_im = im.convert('RGB')
            file_path = os.path.join(UPLOAD_FOLDER, image_uuid) + '.jpg'
            rgb_im.save(file_path)

        # prelogits for user photo
        my_prelogits = extract_prelogits_fv(file_path)
        # find nearest neighbors for each taxon in butterflies
        neighbors = []
        for taxon in butterflies:
            distances = [euclidean(prelogits, my_prelogits) for prelogits in taxon.tp_prelogits]
            closest_idx = np.argmin(distances)
            closest_distance = np.min(distances)
            neighbor = Neighbor(taxon, closest_distance, taxon.tp_photoids[closest_idx])
            neighbors.append(neighbor)
           
        return render_template(
            'nearest.html', 
            user_photo = image_uuid+extension,
            neighbors = neighbors,
        )
    else:
        return render_template('home.html')

@app.route('/tp/')
def render_all_tps():
    return render_template(
        'all_tp_photos.html',
        taxa = butterflies
    )

@app.route('/taxon_photos/<path:path>')
def send_photo(path):
    return send_from_directory('taxon_photos', path)

@app.route('/static/<path:path>')
def send_user_photo(path):
    return send_from_directory('static', path)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6006)
