import tensorflow as tf
import cv2
import time
import argparse

import posenet
from lp_utils.utils import Apng, Stage1Transormator, process_frame, do_projection

parser = argparse.ArgumentParser()

parser.add_argument('--corners_file', type=str, help="Path to a file containing"
                                                     " integer x and y coordinates of the projectable region"
                                                     " inside a web cam frame.\n"
                                                     " Example is provided: corners_example.txt")
parser.add_argument('--apng', type=str, help="Path to a file with a list"
                                             "of the apng animation frames paths.")
parser.add_argument('--proj_width', type=int)
parser.add_argument('--proj_height', type=int)

args = parser.parse_args()


def get_corners(args):
    """
    :return: Tuple of four int pairs denoting corners coordinates
     (Upper left, upper right, down left, down right)

    """

    def to_int_tuple(line):
        x, y = line.split()
        return int(x), int(y)

    path = args.corners_file
    with open(path) as f:
        ul, ur, dl, dr = [to_int_tuple(line)
                          for line in f.readlines()
                          if line.strip()]
    return ul, ur, dl, dr


def proj_dim(args):
    """
    :return: Dimensions of a projecting device
    """
    return args.proj_width, args.proj_height


def get_apng(args):
    """
    :return: List of the animation frames
    """
    path = args.apng
    with open(path) as f:
        frame_paths = [l.strip() for l in f.readlines()]
        frames = [cv2.imread(path, cv2.IMREAD_UNCHANGED)
                for path in frame_paths if path]
        return Apng(frames)


def main():
    apng = get_apng(args)
    corners = get_corners(args)
    proj_dimensions = proj_dim(args)
    try:
        with tf.Session() as sess:
            model_cfg, model_outputs = posenet.load_model(101, sess)
            output_stride = model_cfg['output_stride']
            trans = Stage1Transormator(*corners, *proj_dimensions)
            cap = cv2.VideoCapture(0)
            start = time.time()
            frame_count = 0

            cv2.namedWindow("posenet", cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty("posenet", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            while True:
                _, img = cap.read()
                overlay_image = process_frame(img, apng.next_frame(), trans,
                                              do_projection, sess,
                                              output_stride, model_outputs)
                cv2.imshow('posenet', overlay_image)
                frame_count += 1
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            print('Average FPS: ', frame_count / (time.time() - start))
    finally:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
