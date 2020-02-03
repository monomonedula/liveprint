from abc import ABC, abstractmethod
from typing import List, Tuple, Union


class Poses(ABC):
    @abstractmethod
    def torso_keypoints(self, threshold=0.15) -> Union[List["TorsoKeypoints"], Tuple["TorsoKeypoints"]]:
        pass


class TorsoKeypoints(ABC):
    @property
    @abstractmethod
    def left_shoulder(self) -> "Keypoint":
        pass

    @property
    @abstractmethod
    def right_shoulder(self) -> "Keypoint":
        pass

    @property
    @abstractmethod
    def left_hip(self) -> "Keypoint":
        pass

    @property
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
