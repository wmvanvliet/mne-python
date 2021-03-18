from mne.preprocessing import fix_grad_artifact
import mne

data_path = "/Users/wmvan/data/epasana/lukeminen0058.vhdr"

raw = mne.io.read_raw_brainvision(data_path,
                                  eog=['Eog1', 'Eog2','Eog3', 'Ekg'])
raw.crop(0, 300)
raw.load_data()

raw.filter(0.5, None)

events, event_id = mne.events_from_annotations(raw)

slice_duration = mne.preprocessing.estimate_slice_duration(events, slice_event_id=1128)
slices_per_volume = 21

##
raw_clean = fix_grad_artifact(
    raw,
    n_iter = 100_000,
    n_cascades = 10_000,
    slice_duration = slice_duration,
    TR = slice_duration * slices_per_volume,
    picks = 'eeg',
    copy = True,
    n_jobs = 1,
    verbose = True,
)

raw_clean.plot(decim=4, scalings=dict(eeg=50E-6))
