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

import numpy as np

import logging as log
import paho.mqtt.client as mqtt

from argparse import ArgumentParser
from inference import Network

import utils
import metrics
from transitions import Machine
from frameAnalysisStateMachine import frmAnalyisSM

from utils import release , drawBBoxes


# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60

frameProcessor = frmAnalyisSM()
machine = Machine(frameProcessor, ['PersonIn', 'PersonOut'], initial='PersonOut')
machine.add_transition('newPersonEntered', 'PersonOut', 'PersonIn', before='durationStats')


def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    return parser


def connect_mqtt():
    # Connect to the MQTT client

    client =  mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)

    return client

def infer_on_stream(args, client):

    isImage = False

    # Handle the input stream 
    if args.input != 'CAM':
        assert os.path.isfile(args.input)
    

    if args.input == 'CAM':
        args.input = 0
    elif args.input.endswith(('.jpg', '.bmp', '.png')):
        isImage = True


    last_count = 0
    total_count = 0
    durationList = []
    inferenceList = []
    f_n = 0 #False Negatives for analysis purpose...

    # Initialise the class
    infer_network = Network()

    # Set Probability threshold for detections
    prob_threshold = args.prob_threshold

    # Load the model through `infer_network`
    mdl_start = cv2.getTickCount() 
   
    infer_network.load_model(args.model, args.device, args.cpu_extension)
    load_time = utils.timeLapse(mdl_start)
    
    cap = cv2.VideoCapture(args.input)
    cap.open(args.input)

    if not cap.isOpened():
        log.error("ERROR! Unable to open video source")
        exit(1)

    w,h = utils.getSrcDim(cap) #dimensions from the captured source

    if not isImage:
        out = cv2.VideoWriter('out.mp4',utils.getCODEC(), cap.get(cv2.CAP_PROP_FPS), (w,h))
    else:
        out = None



    #Loop until stream is over
    while cap.isOpened():
        # Read from the video capture 
        flag, frame = cap.read()
        if not flag:
            break
        key_pressed = cv2.waitKey(60)

        # Pre-process the image as needed 
        p_frame = utils.preprocessed_input(infer_network,frame)

        # Start asynchronous inference for specified request 
        inf_start = time.time()
        infer_network.exec_net(p_frame,request_id=0)

        # Wait for the result 
        if infer_network.wait(request_id=0) == 0:
            det_time = time.time() - inf_start
            inferenceList.append(det_time * 1000)

            # Get the results of the inference request 
            result = infer_network.get_output(request_id=0)
           
            # Extract any desired stats from the results 
            frame, count , f_n = drawBBoxes(frame, result, prob_threshold, w, h,last_count,f_n)
          

            # Calculate and send relevant information on 
            inf_time_message = "Inference time: {:.3f}ms".format(det_time * 1000)
            cv2.putText(frame, inf_time_message, (15, 15),cv2.FONT_HERSHEY_COMPLEX, 0.5, (200, 10, 10), 1)

            # When new person enters the video
            if count > last_count:

                frameProcessor.to_PersonOut()   # re-assuring by Setting Previous person to have moved out.
                frameProcessor.newPersonEntered()    #Set the state as new person entered and initialize timer.
                total_count = total_count + count - last_count
                client.publish("person", json.dumps({"total": total_count}))

                ''' 
                log.info('Entered Scene at MS {}'.format(cap.get(cv2.CAP_PROP_POS_MSEC)))
                log.info('Entered Scene at FRM {}'.format(cap.get(cv2.CAP_PROP_POS_FRAMES)))
                log.info('Entered Scene at AVI {}'.format(cap.get(cv2.CAP_PROP_POS_AVI_RATIO)))
                log.info('Frame Rate {}'.format(cap.get(cv2.CAP_PROP_FPS)))
                
                ## The logic of calculating duration with above CV2 attribs worked fine.
                ##  But realised it may not work in CAM mode.. so need to build a generic logic.
                '''

            # current_count, total_count and duration to the MQTT server 
            # Person duration in the video is calculated

            # Topic "person": keys of "count" and "total" 
            # Topic "person/duration": key of "duration" 
            if count < last_count:
                duration = float(time.time() - frameProcessor.getPersonEntrytime())
                frameProcessor.to_PersonOut()
                durationList.append(duration)

                # Publish average duration spent by people to the MQTT server
                client.publish("person/duration", json.dumps({"duration": round(np.mean(durationList)) }))

            client.publish("person", json.dumps({"count": count}))
            last_count = count

            
            if key_pressed == 27:
                break

        # Send the frame to the FFMPEG server 
        sys.stdout.buffer.write(frame)  
        sys.stdout.flush()

        # Write an output image if `single_image_mode` 
        if isImage:
            cv2.imwrite('output/output_image.jpg', frame)
        else:
            out.write(frame)



    
    log.info('######################################################')
    log.info('# Average Inference Time                                             ::  {:.3f} ms'.format(np.mean(inferenceList)))
    log.info('# (IR) Model Size   (XML)                                            ::  {}'.format(metrics.getSize(utils.getMOFiles(args.model)['model'])))
    log.info('# (IR) Model Weight (BIN)                                            ::  {}'.format(metrics.getSize(utils.getMOFiles(args.model)['weights'])))
    log.info('# Total Model Load Time                                              ::  {:.3f} ms'.format(load_time))
    log.info('# Set Probability Threshold                                          ::  {}'.format(prob_threshold))
    log.info('# No. of False Negatives @ 0.75 & 0.5 times of the set threhold      ::  {}'.format(f_n))
    log.info('# Error_percent in detecting Total ppl                               ::  {}'.format(metrics.getErrorPercent(total_count,"people")))
    log.info('# Error_percent in average duration                                  ::  {}'.format(metrics.getErrorPercent(round(np.mean(durationList)),"duration")))
    log.info('######################################################')
    


    release(out,cap,client)



def main():
    """
    Load the network and parse the output.

    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()
    # Connect to the MQTT server
    client = connect_mqtt()
    # Perform inference on the input stream
    infer_on_stream(args, client)

  


if __name__ == '__main__':
    main()
