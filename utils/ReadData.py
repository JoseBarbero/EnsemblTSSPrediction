import numpy as np
import pandas as pd
import os
import re
from Bio.Seq import Seq
from datetime import datetime

def seqfile_to_instances(seqfile):
    with open(seqfile, 'r') as _seqfile:
        return np.array(_seqfile.read().split('\n')[:-1])