"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


import os
import sys
import time
import socket
import json
import cv2
import utils

import numpy as np

import logging as log


from argparse import ArgumentParser


def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
  
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file")

    parser.add_argument("-l", "--light", required=True, type=str,
                        help=" generate vide with lights off / on / exposed")

    return parser


def adjust_gamma(frame, gamma=1.0):

   invGamma = 1.0 / gamma
   table = np.array([((i / 255.0) ** invGamma) * 255
      for i in np.arange(0, 256)]).astype("uint8")

   return cv2.LUT(frame, table)


def adjustLighting(args):
    
    cap = cv2.VideoCapture(args.input)
    cap.open(args.input)
    w,h = utils.getSrcDim(cap) #dimensions from the captured source

    if not cap.isOpened():
        log.error("ERROR! Unable to open video source")
        exit(1)

        
    print(args.light)
    out = cv2.VideoWriter('Pedestrian_Detect_2_1_1.mp4'.format(args.light),utils.getCODEC(), cap.get(cv2.CAP_PROP_FPS), (w,h))


    #Loop until stream is over
    while cap.isOpened():
        # Read from the video capture 
        flag, frame = cap.read()
        if not flag:
            break
        key_pressed = cv2.waitKey(60)

        
        if (args.light is "off"):
            gamma = 0.1
        elif(args.light is "on"):
           gamma = 1.5
        elif(args.light is "exposed"):
            gamma = 2.5
        

        
        newframe = adjust_gamma(frame, gamma=gamma)
    

        if key_pressed == 27:
                break


    
        out.write(newframe)
    
    if out is not None:
        out.release()

    cap.release()
    cv2.destroyAllWindows()





def main():
    """
    Load the network and parse the output.

    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()

    # Perform inference on the input stream
    adjustLighting(args)

  


if __name__ == '__main__':
    main()
