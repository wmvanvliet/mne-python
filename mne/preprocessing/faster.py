import numpy as np
from collections import defaultdict
from scipy.signal import welch, lfilter
from scipy.stats import kurtosis
from . import find_outliers
from ..utils import logger
from ..io.pick import pick_types, channel_type


def _by_ch_type(info, picks):
    """Returns lists of channel indices, grouped by channel type."""
    ch_types = defaultdict(list)

    for ch in picks:
        ch_types[channel_type(info, ch)].append(ch)

    return ch_types.items()


def _hurst(x):
    """Estimate Hurst exponent on a timeseries.

    The estimation is based on the second order discrete derivative.

    Parameters
    ----------
    x : 1D numpy array
        The timeseries to estimate the Hurst exponent for.

    Returns
    -------
    h : float
        The estimation of the Hurst exponent for the given timeseries.
    """
    y = np.cumsum(np.diff(x, axis=1), axis=1)

    b1 = [1, -2, 1]
    b2 = [1,  0, -2, 0, 1]

    # second order derivative
    y1 = lfilter(b1, 1, y, axis=1)
    y1 = y1[:, len(b1) - 1:-1]  # first values contain filter artifacts

    # wider second order derivative
    y2 = lfilter(b2, 1, y, axis=1)
    y2 = y2[:, len(b2) - 1:-1]  # first values contain filter artifacts

    s1 = np.mean(y1 ** 2, axis=1)
    s2 = np.mean(y2 ** 2, axis=1)

    return 0.5 * np.log2(s2 / s1)


def _efficient_welch(data, sfreq):
    """Calls scipy.signal.welch with parameters optimized for greatest speed
    at the expense of precision. The window is set to ~10 seconds and windows
    are non-overlapping.

    Parameters
    ----------
    data : N-D numpy array
        The timeseries to estimate signal power for. The last dimension
        is presumed to be time.
    sfreq : float
        The sample rate of the timeseries.

    Returns
    -------
    fs : 1D numpy array
        The frequencies for which the power spectra was calculated.
    ps : ND numpy array
        The power spectra for each timeseries.
    """
    nperseg = min(data.shape[-1],
                  2 ** int(np.log2(10 * sfreq) + 1))  # next power of 2

    return welch(data, sfreq, nperseg=nperseg, noverlap=0, axis=-1)


def _freqs_power(data, sfreq, freqs):
    """Estimate signal power at specific frequencies.

    Parameters
    ----------
    data : N-D numpy array
        The timeseries to estimate signal power for. The last dimension
        is presumed to be time.
    sfreq : float
        The sample rate of the timeseries.
    freqs : list of float
        The frequencies to estimate signal power for.

    Returns
    -------
    ps : list of float
        For each requested frequency, the estimated signal power.
    """
    fs, ps = _efficient_welch(data, sfreq)
    try:
        return np.sum([ps[..., np.searchsorted(fs, f)] for f in freqs], axis=0)
    except IndexError:
        raise ValueError(
            ("Insufficient sample rate to  estimate power at {} Hz for line "
             "noise detection. Use the 'metrics' parameter to disable the "
             "'line_noise' metric.").format(freqs))


def _power_gradient(data, sfreq, prange):
    """Estimate the gradient of the power spectrum at upper frequencies.

    Parameters
    ----------
    data : N-D numpy array
        The timeseries to estimate signal power for. The last dimension
        is presumed to be time.
    sfreq : float
        The sample rate of the timeseries.
    prange : pair of floats
        The (lower, upper) frequency limits of the power spectrum to use. In
        the FASTER paper, they set these to the passband of the lowpass filter.

    Returns
    -------
    grad : N-D numpy array
        The gradients of each timeseries.
    """
    fs, ps = _efficient_welch(data, sfreq)

    # Limit power spectrum to upper frequencies
    start, stop = (np.searchsorted(fs, p) for p in prange)
    if start >= ps.shape[1]:
        raise ValueError(("Sample rate insufficient to estimate {} Hz power. "
                          "Use the 'power_gradient_range' parameter to tweak "
                          "the tested frequencies for this metric or use the "
                          "'metrics' parameter to disable the "
                          "'power_gradient' metric.").format(prange[0]))
    ps = ps[:, start:stop]

    # Compute mean gradients
    return np.mean(np.diff(ps), axis=1)


def _deviation(data):
    """Computes the deviation from mean for each channel in a set of epochs.

    This is not implemented as a lambda function, because the channel means
    should be cached during the computation.

    Parameters
    ----------
    data : 3D numpy array
        The epochs (#epochs x #channels x #samples).

    Returns
    -------
    dev : 1D numpy array
        For each epoch, the mean deviation of the channels.
    """
    ch_mean = np.mean(data, axis=2)
    return ch_mean - np.mean(ch_mean, axis=0)


def faster_bad_channels(epochs, picks=None, thres=3, use_metrics=None,
                        max_iter=2, return_by_metric=False):
    """Implements the first step of the FASTER algorithm.

    This function attempts to automatically mark bad EEG channels by performing
    outlier detection. It operated on epoched data, to make sure only relevant
    data is analyzed.

    Parameters
    ----------
    epochs : Instance of Epochs
        The epochs for which bad channels need to be marked
    picks : list of int | None
        Channels to operate on. Defaults to EEG channels.
    thres : float
        The threshold value, in standard deviations, to apply. A channel
        crossing this threshold value is marked as bad. Defaults to 3.
    use_metrics : list of str
        List of metrics to use. Can be any combination of:
            'variance', 'correlation', 'hurst', 'kurtosis', 'line_noise'
        Defaults to all of them.
    max_iter : int
        The maximum number of iterations performed during outlier detection
        (defaults to 2, original FASTER paper uses 1).
    return_by_metric : bool
        Whether to return the bad channels as a flat list (False, default) or
        as a dictionary with the names of the used metrics as keys and the
        bad channels found by this metric as values.

    Returns
    -------
    bads : list of str
        The names of the bad EEG channels.
    """
    metrics = {
        'variance': lambda x: np.var(x, axis=1),
        'correlation': lambda x: np.mean(
            np.ma.masked_array(np.corrcoef(x),
                               np.identity(len(x), dtype=bool)), axis=0),
        'hurst': lambda x: _hurst(x),
        'kurtosis': lambda x: kurtosis(x, axis=1),
        'line_noise': lambda x: _freqs_power(x, epochs.info['sfreq'],
                                             [50, 60]),
    }

    if picks is None:
        picks = pick_types(epochs.info, meg=True, eeg=True, exclude=[])
    if use_metrics is None:
        use_metrics = metrics.keys()

    # Concatenate epochs in time
    data = epochs.get_data()
    data = data.transpose(1, 0, 2).reshape(data.shape[1], -1)
    data = data[picks]

    # Find bad channels
    bads = defaultdict(list)
    for ch_type, chs in _by_ch_type(epochs.info, picks):
        logger.info('Bad channel detection on %s channels:' % ch_type.upper())
        for m in use_metrics:
            s = metrics[m](data[chs])
            b = [epochs.ch_names[picks[chs[i]]]
                 for i in find_outliers(s, thres, max_iter)]
            logger.info('\tBad by %s: %s' % (m, b))
            bads[m].append(b)

    bads = dict([(k, np.concatenate(v).tolist()) for k, v in bads.items()])
    if return_by_metric:
        return bads
    else:
        return np.unique(np.concatenate(list(bads.values()))).tolist()


def faster_bad_epochs(epochs, picks=None, thres=3, use_metrics=None,
                      max_iter=2, return_by_metric=False):
    """Implements the second step of the FASTER algorithm.

    This function attempts to automatically mark bad epochs by performing
    outlier detection.

    Parameters
    ----------
    epochs : Instance of Epochs
        The epochs to analyze.
    picks : list of int | None
        Channels to operate on. Defaults to EEG channels.
    thres : float
        The threshold value, in standard deviations, to apply. An epoch
        crossing this threshold value is marked as bad. Defaults to 3.
    use_metrics : list of str
        List of metrics to use. Can be any combination of:
            'amplitude', 'variance', 'deviation'
        Defaults to all of them.
    max_iter : int
        The maximum number of iterations performed during outlier detection
        (defaults to 2, original FASTER paper uses 1).
    return_by_metric : bool
        Whether to return the bad epochs as a flat list (False, default) or
        as a dictionary with the names of the used metrics as keys and the
        bad epochs found by this metric as values.

    Returns
    -------
    bads : list of int
        The indices of the bad epochs.
    """

    metrics = {
        'amplitude': lambda x: np.mean(np.ptp(x, axis=2), axis=1),
        'deviation': lambda x: np.mean(_deviation(x), axis=1),
        'variance': lambda x: np.mean(np.var(x, axis=2), axis=1),
    }

    if picks is None:
        picks = pick_types(epochs.info, meg=True, eeg=True, exclude='bads')
    if use_metrics is None:
        use_metrics = metrics.keys()

    data = epochs.get_data()

    bads = defaultdict(list)
    for ch_type, chs in _by_ch_type(epochs.info, picks):
        logger.info('Bad epoch detection on %s channels:' % ch_type.upper())
        for m in use_metrics:
            s = metrics[m](data[:, chs])
            b = find_outliers(s, thres, max_iter)
            logger.info('\tBad by %s: %s' % (m, b))
            bads[m].append(b)

    bads = {k: np.concatenate(v).tolist() for k, v in bads.items()}
    if return_by_metric:
        return bads
    else:
        return np.unique(np.concatenate(list(bads.values()))).tolist()


def faster_bad_components(ica, epochs, thres=3, use_metrics=None,
                          power_gradient_range=None, max_iter=2,
                          return_by_metric=False):
    """Implements the third step of the FASTER algorithm.

    This function attempts to automatically mark bad ICA components by
    performing outlier detection.

    Parameters
    ----------
    ica : Instance of ICA
        The ICA operator, already fitted to the supplied Epochs object.
    epochs : Instance of Epochs
        The untransformed epochs to analyze.
    thres : float
        The threshold value, in standard deviations, to apply. A component
        crossing this threshold value is marked as bad. Defaults to 3.
    use_metrics : list of str
        List of metrics to use. Can be any combination of:
            'eog_correlation', 'kurtosis', 'power_gradient', 'hurst',
            'median_gradient'
        Defaults to all of them.
    power_gradient_range : pair of floats
        The (lower, upper) frequency limits of the power spectrum to use when
        calculating the 'power_gradient' metric. In the FASTER paper, they set
        these to the passband of the lowpass filter. Defaults to (25, 40).
    max_iter : int
        The maximum number of iterations performed during outlier detection
        (defaults to 2, original FASTER paper uses 1).
    return_by_metric : bool
        Whether to return the bad components as a flat list (False, default) or
        as a dictionary with the names of the used metrics as keys and the
        bad components found by this metric as values.

    Returns
    -------
    bads : list of int
        The indices of the bad components.

    See also
    --------
    ICA.find_bads_ecg
    ICA.find_bads_eog
    """
    if power_gradient_range is None:
        power_gradient_range = (25., 40.)

    epochs = epochs.pick_channels(ica.info['ch_names'])
    source_data = ica.get_sources(epochs).get_data().transpose(1, 0, 2)
    source_data = source_data.reshape(source_data.shape[0], -1)

    # Compute the transform matrix used by the ICA operator is necessary
    if use_metrics is None or 'kurtosis' in use_metrics:
        transform_matrix = np.dot(ica.mixing_matrix_.T,
                                  ica.pca_components_[:ica.n_components_])
        if hasattr(ica, '_pre_whitener') and ica.noise_cov is not None:
            transform_matrix = np.dot(transform_matrix, ica._pre_whitener)

    metrics = {
        'eog_correlation': lambda x: x.find_bads_eog(epochs)[1],
        'kurtosis': lambda x: kurtosis(transform_matrix, axis=1),
        'power_gradient': lambda x: _power_gradient(source_data,
                                                    x.info['sfreq'],
                                                    power_gradient_range),
        'hurst': lambda x: _hurst(source_data),
        'median_gradient': lambda x: np.median(np.abs(np.diff(source_data)),
                                               axis=1),
        'line_noise': lambda x: _freqs_power(source_data,
                                             epochs.info['sfreq'],
                                             [50, 60]),
    }

    if use_metrics is None:
        use_metrics = metrics.keys()
        if 'eog' not in epochs:
            use_metrics = list(use_metrics)
            use_metrics.remove('eog_correlation')

    bads = dict()
    for m in use_metrics:
        scores = np.atleast_2d(metrics[m](ica))
        for s in scores:
            b = find_outliers(s, thres, max_iter)
            logger.info('Bad by %s:\n\t%s' % (m, b))
            bads[m] = b

    if return_by_metric:
        return bads
    else:
        return np.unique(np.concatenate(list(bads.values()))).tolist()


def faster_bad_channels_in_epochs(epochs, picks=None, thres=3,
                                  use_metrics=None, max_iter=2,
                                  return_by_metric=False):
    """Implements the fourth step of the FASTER algorithm.

    This function attempts to automatically mark bad channels in each epochs by
    performing outlier detection.

    Parameters
    ----------
    epochs : Instance of Epochs
        The epochs to analyze.
    picks : list of int | None
        Channels to operate on. Defaults to EEG channels.
    thres : float
        The threshold value, in standard deviations, to apply. An epoch
        crossing this threshold value is marked as bad. Defaults to 3.
    use_metrics : list of str
        List of metrics to use. Can be any combination of:
            'amplitude', 'variance', 'deviation', 'median_gradient'
        Defaults to all of them.
    max_iter : int
        The maximum number of iterations performed during outlier detection
        (defaults to 2, original FASTER paper uses 1).
    return_by_metric : bool
        Whether to return the bad channels as a nested list (False, default) or
        as a dictionary with the names of the used metrics as keys and the
        bad channels found by this metric as a nested list.

    Returns
    -------
    bads : list of lists of int
        For each epoch, the indices of the bad channels.
    """

    metrics = {
        'amplitude': lambda x: np.ptp(x, axis=2),
        'deviation': lambda x: _deviation(x),
        'variance': lambda x: np.var(x, axis=2),
        'median_gradient': lambda x: np.median(np.abs(np.diff(x)), axis=2),
        'line_noise': lambda x: _freqs_power(x, epochs.info['sfreq'],
                                             [50, 60]),
    }

    if picks is None:
        picks = pick_types(epochs.info, meg=False, eeg=True, exclude='bads')
    if use_metrics is None:
        use_metrics = metrics.keys()

    data = epochs.get_data()

    bads = defaultdict(lambda: [list() for _ in range(len(epochs))])
    for ch_type, chs in _by_ch_type(epochs.info, picks):
        for m in use_metrics:
            logger.info('Bad channel-in-epoch detection on %s channels:'
                        % ch_type.upper())
            s_epochs = metrics[m](data[:, chs])
            for i, s in enumerate(s_epochs):
                b = [epochs.ch_names[picks[j]]
                     for j in find_outliers(s, thres, max_iter)]
                if len(b) > 0:
                    logger.info('Epoch %d, Bad by %s:\n\t%s' % (i, m, b))
                bads[m][i].append(b)

    bads = {k: sum(v, []) for k, v in bads.items()}
    if return_by_metric:
        return dict(bads)
    else:
        bads = [np.unique(np.concatenate(b_)).tolist()
                for b_ in bads.values() if len(b_) > 0]
        return bads
