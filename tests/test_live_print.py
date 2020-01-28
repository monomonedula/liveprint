import os
import pathlib

import cv2
import numpy as np

from liveprint import Projector, Background, get_apng
from lp_utils.pose import Keypoint
from lp_utils.utils import Apng


class FakePosesFactory:
    def poses(self, image):
        return FakePoses()


class FakePoses:
    def torso_keypoints(self, threshold=0.15):
        return iter([FakeTorsoKeypoints()])


class FakeTorsoKeypoints:
    def __init__(
        self,
        left_shoulder=(740, 161,),
        right_shoulder=(875, 150,),
        left_hip=(759, 308,),
        right_hip=(862, 311,),
    ):
        self.left_shoulder = Keypoint(5, *left_shoulder, 0.6)
        self.right_shoulder = Keypoint(6, *right_shoulder, 0.6)
        self.left_hip = Keypoint(11, *left_hip, 0.6)
        self.right_hip = Keypoint(12, *right_hip, 0.6)


class FakeProjectableRegion:
    def __init__(self, output_height=768, output_width=1024):
        self._output_resolution = (output_height, output_width, 3)

    def of(self, webcam_img):
        return 255 * np.ones(shape=self._output_resolution, dtype=np.uint8)


def test_projector():
    path = os.path.join(pathlib.Path(__file__).parent.absolute(), "..", "resources", "test_image_1.png")
    projectable_region_dims = [768, 1024]
    output_image = Projector(
        Background([*projectable_region_dims, 3]),
        FakePosesFactory(),
        FakeProjectableRegion(*projectable_region_dims),
        Apng([cv2.imread(path, cv2.IMREAD_UNCHANGED)])
    ).project(None)
