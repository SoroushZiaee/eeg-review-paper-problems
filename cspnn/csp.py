import itertools
import numpy as np
import scipy.linalg as la


def make_binary(y, key: int):
    idx = y == key
    return idx, ~idx


def calc_cov(EEG_data):
    """
    INPUT:
    EEG_data : EEG_data in shape T x N x S

    OUTPUT:
    avg_cov : covariance matrix of averaged over all trials
    """
    cov = []
    for i in range(EEG_data.shape[0]):
        cov.append(EEG_data[i] @ EEG_data[i].T / np.trace(EEG_data[i] @ EEG_data[i].T))

    cov = np.mean(np.array(cov), 0)

    return cov


class CSP:
    def __init__(self, labels: list, num_bands: int, num_windows: int):
        self.labels = labels
        self.num_bands = num_bands
        self.num_windows = num_windows

        self.template = "label-{}_band-{}_window-{}"

    def calculate_weights(self, signals, labels):
        trials, channels, n_windows, n_bands, data = signals.shape

        assert n_windows == self.num_windows
        assert n_bands == self.num_bands

        w_dict = {}
        for label in self.labels:  # np.unique(labels):
            trial_idx, rest_idx = make_binary(labels, label - 1)

            x_trial = signals[trial_idx]
            x_rest = signals[rest_idx]

            for freq_band in range(self.num_bands):
                for window in range(self.num_windows):
                    # (batch, num_channel, timepoint)
                    R_a = calc_cov(x_trial[:, :, window, freq_band, :])
                    R_b = calc_cov(x_rest[:, :, window, freq_band, :])
                    R_c = R_a + R_b
                    
                    # assert np.allclose(R_a @ R_c, R_c @ R_a)

                    lda, V = la.eig(R_c)
                    W = la.sqrtm(la.inv(np.diag(lda))) @ V.T
                    T_a = W @ R_a @ W.T
                    T_b = W @ R_b @ W.T
                    T_c = W @ R_c @ W.T
                    
                    assert np.allclose(T_a @ T_c, T_c @ T_a) # these matrices must commute inorder
                    # to be simultaniously diagonalizable

                    sai, E = la.eig(T_a, T_c)
                    P = E.real  # or P = E.T @ W although the latter doesn't give
                    # a coordinate system with orthogonal axis. we skip
                    # whitning step
                    # P = E.T @ W
                    # P = P.real

                    sai_a, E_a = la.eig(T_a)
                    sai_b, E_b = la.eig(T_b)
                    assert np.allclose(E_a, E_b)

                    assert np.allclose(
                        np.diag(sai_a) + np.diag(sai_b), np.eye(sai_b.shape[0])
                    )
                    assert np.allclose(E @ E.T, np.eye(E.shape[0]))
                    # assert np.allclose(P @ P.T, np.eye(W.shape[0]))
                    w_dict[self.template.format(label, freq_band, window)] = P

        self.w_dict = w_dict

        return w_dict

    def _calc_feat(self, signal: np.ndarray, weight: np.ndarray):
        """Apply CSP weights to the signal

        Parameters
        ----------
        signal : np.ndarray
            An 2d numpy array with [channels, timepoints]
        weight : np.ndarray
            _description_

        Returns
        -------
        _type_
            _description_
        """
        x = weight.T @ signal

        tmp_element = x @ x.T

        num = np.diag(tmp_element)
        den = np.trace(tmp_element)

        features = np.log(num / den)

        return features

    def calc_feat(self, signals, num_features=None):
        trials, channels, windows, freq_bands, _ = signals.shape
        weights_dim = list(self.w_dict.values())[0].shape[0]
        assert weights_dim == channels
        if num_features is not None:
            assert num_features <= (weights_dim // 2)

            idx = []
            for i in range(num_features):
                idx.append(i)
            for i in reversed(idx):
                idx.append(-(i + 1))
            num_features *= 2
        else:
            num_features = weights_dim
            idx = list(range(num_features))

        features = np.vstack(
            [
                np.expand_dims(
                    np.vstack(
                        [
                            np.expand_dims(
                                self._calc_feat(
                                    signals[trial, :, i[2], i[1], :],
                                    self.w_dict[self.template.format(*i)][:, idx],
                                ),
                                axis=0,
                            )
                            for i in itertools.product(
                                self.labels,
                                range(freq_bands),
                                range(windows),
                            )
                        ]
                    ),
                    axis=0,
                )
                for trial in range(trials)
            ]
        )

        return features
