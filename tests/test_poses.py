import tensorflow as tf
import cv2

import posenet
from liveprint import PosesFactory, WebcamBackground, Projector, Background
from lp_utils.utils import ProjectableRegion, Apng


class ImageBg:
    def __init__(self, img):
        self._img = img

    def layers(self):
        return [self._img]


def test_poses():
    with tf.Session() as sess:
        model_cfg, model_outputs = posenet.load_model(101, sess)
        output_stride = model_cfg["output_stride"]
        image = cv2.imread("resources/people.jpg", cv2.IMREAD_UNCHANGED)
        corners = [(151, 70), (725, 70), (142, 462), (740, 461)]
        proj_dim = [640, 480]
        output = Projector(
            ImageBg(ProjectableRegion(*corners, *proj_dim).of(image)),
            PosesFactory(sess, output_stride, model_outputs),
            ProjectableRegion(*corners, *proj_dim),
            Apng([cv2.imread("resources/test_image_1.png", cv2.IMREAD_UNCHANGED)]),
        ).project(image)

        cv2.imwrite("people_test.jpg", output)
