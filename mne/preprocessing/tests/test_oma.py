from mne.preprocessing import fix_grad_artifact, estimate_slice_duration
import mne

#data_path = "/Users/wmvan/data/epasana/lukeminen0058.vhdr"
data_path = "/m/nbe/scratch/epasana/eeg/3017-EEG-data/lukeminen0058.vhdr"

raw = mne.io.read_raw_brainvision(data_path,
                                  eog=['EOG1', 'EOG2', 'EOG3', 'ECG'],
                                  preload=True)
raw.set_montage('standard_1005')
raw.filter(0.5, None)

events, event_id = mne.events_from_annotations(raw)

slice_duration = estimate_slice_duration(events, slice_event_id=1128)
slices_per_volume = 21

raw_clean = fix_grad_artifact(
    raw,
    n_iter=200_000,
    n_cascades=100,
    slice_duration=slice_duration,
    TR=slice_duration * slices_per_volume,
    picks='eeg',
    copy=True,
    n_jobs=4,
    verbose=True,
)

raw_clean, events = raw_clean.resample(500, events=events)
epochs = mne.Epochs(raw_clean, events, event_id=[2, 4, 8, 16, 32], tmax=0.6)
ev = [epochs[cl].average() for cl in epochs.event_id.keys()]
mne.viz.plot_evoked_topo(ev)
