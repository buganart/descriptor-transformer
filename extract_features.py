#!/usr/bin/env ipython
import sys
from pathlib import Path

import numpy as np
import tqdm

from desc import feature_extraction

paths = list(Path(sys.argv[1]).rglob("*.wav"))
out_path = sys.argv[2]

hop_len = 4620  # 15400 * 3 / 10
data = [feature_extraction.load_wav(path)[0] for path in tqdm.tqdm(paths)]
data = [feature_extraction.extract(x, 44100, hop_len) for x in tqdm.tqdm(data)]
data = np.vstack(data)

np.save(out_path, data)
