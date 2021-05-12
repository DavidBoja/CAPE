
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import argparse
import os
import os.path as osp

import pickle

from tqdm import tqdm
import numpy as np

# poziv
# clean_fn('body_models/smpl/basicmodel_m_lbs_10_207_0_v1.0.0.pkl','body_models/smpl/clean/')
def clean_fn(fn, output_folder='output'):
    with open(fn, 'rb') as body_file:
        body_data = pickle.load(body_file, encoding='latin1')

    output_dict = {}
    for key, data in body_data.iteritems():
        if 'chumpy' in str(type(data)):
            output_dict[key] = np.array(data)
        else:
            output_dict[key] = data

    out_fn = osp.split(fn)[1]

    out_path = osp.join(output_folder, out_fn)
    with open(out_path, 'wb') as out_file:
        pickle.dump(output_dict, out_file)