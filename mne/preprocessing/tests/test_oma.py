#from mne.preprocessing import fix_grad_artifact
import sys
import os 

sys.path.append("/home/sfc/coding/mne-python/mne/")
#import oma
print(sys.path)
import mne.io


from mne.preprocessing import fix_grad_artifact

data_path = "/home/sfc/coding/data/lukeminen0058.vhdr"

# Should be high passed / centered
# raw = mne.io.read_raw_brainvision(data_path,
              eog=['Eog1', 'Eog2','Eog3', 'Ekg'], preload = True)
#raw.plot()
# fix_grad_artifact(raw, 1,  1)
