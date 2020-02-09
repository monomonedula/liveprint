from abc import ABC, abstractmethod
from typing import List, Tuple, Union


class PosesFactory(ABC):
    @abstractmethod
    def poses(self, img) -> "Poses":
        pass


class Poses(ABC):
    @abstractmethod
    def torso_keypoints(
        self, threshold=0.15
    ) -> Union[List["TorsoKeyPoints"], Tuple["TorsoKeyPoints"]]:
        pass


class TorsoKeyPoints(ABC):
    @abstractmethod
    def left_shoulder(self) -> "Keypoint":
        pass

    @abstractmethod
    def right_shoulder(self) -> "Keypoint":
        pass

    @abstractmethod
    def left_hip(self) -> "Keypoint":
        pass

    @abstractmethod
    def right_hip(self) -> "Keypoint":
        pass


class Keypoint(ABC):
    @abstractmethod
    def threshold(self, thresh) -> bool:
        pass

    @abstractmethod
    def coords(self) -> Tuple[int, int]:
        pass
