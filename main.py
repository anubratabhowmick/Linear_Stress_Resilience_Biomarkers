# _*_ coding: utf-8 _*_
# Author: Anubrata Bhowmick
# @Time:

import pathlib
import shutil
import argparse

import pandas as pd

from SubjectSpaceAtlas import SubjectSpaceAtlas
from Biomarkers import Biomarkers

parser = argparse.ArgumentParser()
parser.add_argument('--atlas_creation', type=bool, default=False, help='Create co-registered atlas')
parser.add_argument('--atlas_viz', type=bool, default=False, help='Visualize co-registered atlas image')
parser.add_argument('--classification', type=bool, default=True, help='Start Classification')
parser.add_argument('--viz_mean_matrix', type=bool, default=False, help='Visualize Group Adjacency Matrices')
parser.add_argument('--prepare_dataset', type=bool, default=True, help='Prepare dataset for classification purposes')
parser.add_argument('--save_dataset', type=bool, default=False, help='Save dataset for classification purposes')
parser.add_argument('--find_biomarkers', type=bool, default=True, help='Find biomarkers of brain resilience')
opt = parser.parse_args()

if opt.atlas_creation:
    SubjectSpaceAtlas().create_atlas()
    opt.atlas_creation = False

if opt.atlas_viz:
    SubjectSpaceAtlas().visualize_atlas()
    opt.atlas_viz = False

if opt.classification:
    X = pd.DataFrame()
    Y = pd.Series()
    features = []
    if opt.prepare_dataset:
        X, Y, features = Biomarkers().prepare_dataset(opt.save_dataset)
        # print(type(X))
        # print(type(Y))
        # print(type(features))

    if opt.find_biomarkers:
        metrics = Biomarkers().find_biomarkers_brain_resilience(X, Y, features)
        print(metrics)

if opt.viz_mean_matrix:
    Biomarkers().viz_group_adjacency_matrices()


