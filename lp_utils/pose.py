from typing import List

import posenet


class Poses:
    def __init__(self, pose_scores, keypoint_scores, keypoint_coords):
        self.pose_scores = pose_scores
        self.keypoint_scores = keypoint_scores
        self.keypoint_coords = keypoint_coords

    def torso_keypoints(self, threshold=0.15):
        return tuple(
            TorsoKeypoints(list(Keypoints(kc, ks)))
            for pscore, ks, kc in zip(
                self.pose_scores[self.pose_scores >= threshold],
                self.keypoint_scores,
                self.keypoint_coords,
            )
        )


class TorsoKeypoints:
    names = {
        "left_shoulder": 5,
        "right_shoulder": 6,
        "left_hip": 11,
        "right_hip": 12,
    }

    def __init__(self, kpts: List["Keypoint"]):
        self.kpts = kpts

    def __getattr__(self, name):
        return self.kpts[self.names[name]]


class Keypoints:
    def __init__(self, kpts, kpts_scores):
        self._kpts = kpts
        self._kpts_scores = kpts_scores

    def __iter__(self):
        return iter(
            Keypoint(n, x, y, score)
            for n, ((y, x), score) in enumerate(zip(self._kpts, self._kpts_scores))
        )

    def iter_threshold(self, t):
        return filter(lambda kpt: kpt.threshold(t), iter(self))


class Keypoint:
    def __init__(self, number, x, y, score):
        self.number = number
        self.x = x
        self.y = y
        self.score = score

    def name(self):
        return posenet.PART_NAMES[self.number]

    def threshold(self, thresh):
        return self.score >= thresh

    def coords(self, inv=False):
        return int(self.x), int(self.y)
