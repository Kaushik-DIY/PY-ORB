"""
Rotation-consistency histogram for feature match filtering.
This module keeps only the dominant orientation bins after descriptor matching.
"""

from __future__ import annotations

import numpy as np


# Bin feature-match rotations and keep the dominant orientation modes.
class RotationHistogram(object):
    def __init__(self, histogram_length=12):
        self.histogram_length = int(histogram_length)
        self.factor = float(self.histogram_length) / 360.0
        self.histo = [[] for _ in range(self.histogram_length)]

    def push(self, rot, idx):
        rot = rot % 360.0
        bin_idx = int(round(rot * self.factor))

        if bin_idx == self.histogram_length:
            bin_idx = 0

        assert 0 <= bin_idx < self.histogram_length
        self.histo[bin_idx].append(idx)

    def push_entries(self, rots, idxs):
        rot_array = np.mod(rots, 360.0)
        bins = np.round(rot_array * self.factor).astype(int)
        bins[bins == self.histogram_length] = 0

        if not np.all((bins >= 0) & (bins < self.histogram_length)):
            raise ValueError("RotationHistogram: Invalid bin index in push_entries()")

        for bin_idx, idx in zip(bins, idxs):
            self.histo[int(bin_idx)].append(idx)

    def compute_3_max(self):
        counts = np.array([len(bin_values) for bin_values in self.histo])
        indices = np.argsort(counts)[::-1]
        max1, max2, max3 = indices[:3]

        if counts[max2] < 0.1 * counts[max1]:
            max2 = -1
        if counts[max3] < 0.1 * counts[max1]:
            max3 = -1

        return int(max1), int(max2), int(max3)

    def get_invalid_idxs(self):
        ind1, ind2, ind3 = self.compute_3_max()
        invalid_idxs = []

        for i in range(self.histogram_length):
            if i != ind1 and i != ind2 and i != ind3:
                invalid_idxs.extend(self.histo[i])

        return invalid_idxs

    def get_valid_idxs(self):
        ind1, ind2, ind3 = self.compute_3_max()
        valid_idxs = []

        if ind1 != -1:
            valid_idxs.extend(self.histo[ind1])
        if ind2 != -1:
            valid_idxs.extend(self.histo[ind2])
        if ind3 != -1:
            valid_idxs.extend(self.histo[ind3])

        return valid_idxs

    def __str__(self):
        return "RotationHistogram " + str(self.histo)

    @staticmethod
    def filter_matches_with_histogram_orientation(idxs1, idxs2, angles1, angles2):
        if len(idxs1) == 0 or len(idxs2) == 0:
            return []

        assert len(idxs1) == len(idxs2)

        num_matches = len(idxs1)
        rot_histo = RotationHistogram()

        rots = angles1[idxs1] - angles2[idxs2]
        rot_histo.push_entries(rots, [ii for ii in range(num_matches)])

        return rot_histo.get_valid_idxs()
