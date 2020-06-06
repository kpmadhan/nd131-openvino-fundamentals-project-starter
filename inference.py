#!/usr/bin/env python3
"""
 Copyright (c) 2018 Intel Corporation.

 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit persons to whom the Software is furnished to do so, subject to
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
import logging as log
from openvino.inference_engine import IENetwork, IECore
import cv2
import utils
import logging as log


class Network:
    """
    Load and configure inference plugins for the specified target devices 
    and performs synchronous and asynchronous modes for the specified infer requests.
    """

    def __init__(self):

        #Initializing class variables
        self.ie = None
        self.network = None

        self.input_blob = None
        self.output_blob = None

        self.exec_network = None
        self.infer_request = None
        return 

    def load_model(self, model, device="CPU", cpu_extension=None):
      
        #Loading the model
        log.info('Initiating IE core....')
        startTime = cv2.getTickCount()
        self.ie = IECore()
        self.network = self.ie.read_network(model=utils.getMOFiles(model)['model'], weights=utils.getMOFiles(model)['weights'])
        log.info('Model metadata read Sucessfully')
        
        #Checking for supported layers
        if not utils.isLayersSupported(self.ie,self.network,device):
            log.error('Cannot continue due to unsupported layers. Check if extension exist !!  Exiting....')
            exit(1)

        #Adding any necessary extensions ###
        if cpu_extension and "CPU" in device:
            self.ie.add_extension(cpu_extension, device)

        # Load the IENetwork
        log.info('Initiating Model loading....')
        startTime = cv2.getTickCount()
        self.exec_network = self.ie.load_network(network=self.network, device_name=device,num_requests=0)
        log.info(f'Model Loaded in {(cv2.getTickCount()-startTime)/cv2.getTickFrequency()} seconds')
        
        # Get the input layer
        self.input_blob = next(iter(self.network.inputs))
        log.info("%s", self.input_blob)

        self.output_blob = next(iter(self.network.outputs))
        log.info("%s", self.output_blob)

        # Returning the loaded inference engine
        log.info("IR successfully loaded into Inference Engine.")
        return self.exec_network

    def get_input_shape(self):
        # Returning the shape of the input layer ###
        return self.network.inputs[self.input_blob].shape

    def exec_net(self,image):
        ### TODO: Start an asynchronous request ###
        ### TODO: Return any necessary information ###
        ### Note: You may need to update the function parameters. ###
        return

    def wait(self):
        ### TODO: Wait for the request to be complete. ###
        ### TODO: Return any necessary information ###
        ### Note: You may need to update the function parameters. ###
        return 

    def get_output(self):
        ### TODO: Extract and return the output results
        ### Note: You may need to update the function parameters. ###
        return 
