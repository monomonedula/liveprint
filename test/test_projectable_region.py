import os
import pathlib
from typing import Tuple

import cv2
import numpy as np

from liveprint.system_setup import PRCoords
from liveprint.utils import StaticProjectableRegion


class FakePRCoords(PRCoords):
    def __init__(self, coords):
        self._coords = coords

    def ul(self) -> Tuple[int, int]:
        return self._coords[0]

    def ur(self) -> Tuple[int, int]:
        return self._coords[1]

    def ll(self) -> Tuple[int, int]:
        return self._coords[3]

    def lr(self) -> Tuple[int, int]:
        return self._coords[2]

    def all(
        self,
    ) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int], Tuple[int, int]]:
        return self._coords


def test_pr():
    """
    make sure the projectable region is extracted correctly
    """
    input_path = os.path.join(
        pathlib.Path(__file__).parent.absolute(), "..", "resources", "test_image_2.jpg"
    )
    out = StaticProjectableRegion(
        FakePRCoords(((478, 276), (1158, 265), (1158, 705), (447, 726),)), 1024, 768,
    ).of(cv2.imread(input_path, cv2.IMREAD_COLOR))
    expected_output = cv2.imread(
        os.path.join(
            pathlib.Path(__file__).parent.absolute(),
            "..",
            "resources",
            "test_output_3.jpg",
        ),
        cv2.IMREAD_UNCHANGED,
    )
    np.testing.assert_almost_equal(out, expected_output)
