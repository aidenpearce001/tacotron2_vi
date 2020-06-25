import os
import glob
import time
import sys
sys.path.append('waveglow/')
import numpy as np
import torch

from hparams import create_hparams
from model import Tacotron2
from layers import TacotronSTFT, STFT
from audio_processing import griffin_lim
from train import load_model
from text import text_to_sequence
from denoiser import Denoiser

import flask
from flask import Flask, request,render_template,jsonify

app = Flask(__name__)

path = 'checkpoint*'
ls_file = glob.glob(path)
ls_date = []

for i in ls_file:
    file1 = os.path.getmtime(i)
    ls_date.append(file1)
# print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(min(ls_date))))

if len(ls_date) > 10 : 
    # print(ls_file[ls_date.index(min(ls_date))])
    os.remove(ls_file[ls_date.index(min(ls_date))])
