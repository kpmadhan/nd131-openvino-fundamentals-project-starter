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
      
        # Load the model
        log.debug('Initiating IE core....')
        startTime = cv2.getTickCount()
        self.ie = IECore()
        self.network = self.ie.read_network(model=utils.getMOFiles(model)['model'], weights=utils.getMOFiles(model)['weights'])
        log.debug('Model metadata read Sucessfully')
        
        # Check for supported layers
        if not utils.isLayersSupported(self.ie,self.network,device):
            log.error('Cannot continue due to unsupported layers. Check if extension exist !!  Exiting....')
            exit(1)

        # Add necessary extensions 
        if cpu_extension and "CPU" in device:
            self.ie.add_extension(cpu_extension, device)

        # Load the IENetwork
        log.debug('Initiating Model loading....')
        startTime = cv2.getTickCount()
        self.exec_network = self.ie.load_network(network=self.network, device_name=device,num_requests=0)
        log.debug('Model Loaded in %s seconds' , utils.timeLapse(startTime))
        
        # Get the input layer
        self.input_blob = next(iter(self.network.inputs))
        log.debug("%s", self.input_blob)

        self.output_blob = next(iter(self.network.outputs))
        log.debug("%s", self.output_blob)

        # Return the loaded inference engine
        log.debug("IR successfully loaded into Inference Engine.")
        return self.exec_network

    def get_input_shape(self):

        # Return the shape of the input layer
        log.debug("Returning input shape")
        return self.network.inputs[self.input_blob].shape

    def exec_net(self,image,request_id):

        # Start an asynchronous request
        startTime = cv2.getTickCount()
        self.exec_network.start_async(request_id=request_id, inputs={self.input_blob: image})
        log.debug('Async request started. Time lapse :: %s seconds' , {utils.timeLapse(startTime)})
        
        return

    def wait(self,request_id):

        # Wait for the request to be complete
        startTime = cv2.getTickCount()
        status = self.exec_network.requests[request_id].wait(-1)
        log.debug('Waiting for the reuest to complete. Time lapse :: %s seconds' , {utils.timeLapse(startTime)})

        return status 

    def get_output(self,request_id):

        # Extract and return the output results
        log.debug("Returning out")
        return self.exec_network.requests[request_id].outputs[self.output_blob] 
