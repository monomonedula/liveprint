import tensorflow as tf
import cv2
import time
import argparse

import posenet
from lp_utils.utils import Apng, Stage1Transormator, process_frame, do_projection

ul = 151, 70
ur = 525, 70
dl = 142, 362
dr = 540, 361


def main():
    try:
        with tf.Session() as sess:
            model_cfg, model_outputs = posenet.load_model(101, sess)
            output_stride = model_cfg['output_stride']

            paths = ['apng/oldschool to ae{:03d}.png'.format(i) for i in range(111)]
            frames = [cv2.imread(p, cv2.IMREAD_UNCHANGED) for p in paths]
            apng = Apng(frames)
            trans = Stage1Transormator(ul, ur, dl, dr, 1024, 768)
            cap = cv2.VideoCapture(0)
            start = time.time()
            frame_count = 0

            cv2.namedWindow("posenet", cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty("posenet", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            while True:
                # print(1)
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
