from ..utils import verbose, _check_preload, logger

import numpy as np


@verbose
def fix_grad_artifact(raw, n_iter, n_cascades, slice_duration='auto',
                      TR='auto', picks='eeg', copy=True, n_jobs=1,
                      verbose=True):
    """Remove fMRI gradient artifact using OMA filter.

    Use the Optimized Moving Average algorithm to remove the gradient artifact
    from EEG data -- a very prominent artifact occuring during concurrent
    measures of EEG and (f)MRI data as detailed in
    :footcite:`FerreiraEtAl2016`.

    Parameters
    ----------
    raw : Raw
        The Raw EEG data we want to filter. The data need to be preloaded.
    n_iter : int
        The number of iterations of the filter. The more iteration, the
        tighter the filter. Defaults to [FIXME]
    n_cascades : int
        The number of cascades of the filter. Defaults to [FIXME]
    slice_duration : float | 'auto'
        Slice duration in samples - default to Auto. Note that this doesn't
        need to be an integer value.
    TR : float | 'auto'
        Repetition time between two volumes in seconds - defaults to Auto.
    %(picks_base)s all EEG channels.
    copy : bool
        Wether to make a copy of the data or operate in place.
        Defaults to True.
    %(n_jobs)s
    %(verbose)s

    Returns
    -------
    raw_filt : Raw
        The filtered instance of the data.

    References
    ----------
    .. footbibliography::

    Notes
    -----
    Make sure there are no slow drifts in the signal before using this
    function. Slow drifts may be removed from example by highpass filtering the
    data.

    See also
    --------
    estimate_slice_duration
    """
    _check_preload(raw, 'fix_grad_artifact')
    if copy:
        raw = raw.copy()

    def OMA(M):
        # The filter is described in formula 12, 15 and 16 in the paper.  In
        # order to transcribe these formulas directly into python code, we set
        # up the variables used in the formulas first.
        N = len(raw.times)
        k = np.arange(N)
        z = np.exp(1j * 2 * np.pi * k / N)
        J = n_iter
        L = n_cascades

        # Formula 12
        filt = (1 / M**2) * (1 - z**(-M)) * (1 - z**M) / ((1 - z**(-1)) * (1 - z))  # noqa
        filt[z == 1 + 0j] = 0  # fix divide by zero cases

        # Formula 15
        filt = 1 - (1 - filt) ** J

        # Formula 16
        filt = filt ** L
        return filt

    logger.info(f'Computing OMA filter using {n_iter} iterations and '
                f'{n_cascades} cascades...')
    filt = OMA(slice_duration)
    filt *= OMA(TR)

    def apply_filter(signal):
        """Apply the filter to a single channel."""
        signal_fft = np.fft.fft(signal)
        signal_fft = filt * signal_fft
        return np.fft.ifft(signal_fft)

    raw.apply_function(apply_filter, picks=picks, n_jobs=n_jobs,
                       verbose=verbose)
    return raw


def estimate_slice_duration(events, slice_event_id):
    """Estimate the slice duration from slice onset events.

    First, the time differences (=deltas) between consecutive events are
    computed. Then, the median delta is taken as rough estimate of the slice
    duration. Next, all deltas that are within one sample of the median delta
    are assumed to be valid slice durations. The mean of the slice durations is
    used as final estimate.

    Parameters
    ----------
    events : ndarray, shape (n_events, 3)
        The events as for example produced by :func:`mne.find_events`.
    slice_event_id : int
        The event code marking when a new slice begins.

    Returns
    -------
    slice_duration : float
        An estimate of the slice duration in samples.

    Notes
    -----
    This function will fail if the median time difference between consecutive
    slice events is not a good estimate of the slice duration.

    See also
    --------
    fix_grad_artifact
    """
    onsets = events[events[:, 2] == slice_event_id][:, 0]
    deltas = np.diff(onsets)
    slice_durations = deltas[np.abs(deltas - np.median(deltas)) <= 1]
    return slice_durations.mean()
