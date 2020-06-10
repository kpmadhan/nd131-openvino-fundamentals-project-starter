import os
import logging as log
import cv2
import sys

def getMOFiles(model):
    d = dict()
    d['model'] = model
    d['weights']   = os.path.splitext(model)[0] + ".bin"
    return d 


def isLayersSupported(ie, network,device):
    supported_layers = ie.query_network(network, device_name=device)
    unsupported_layers = [l for l in network.layers.keys() if l not in supported_layers]
    if len(unsupported_layers) != 0:
        log.info("Unsupported layers found: {}".format(unsupported_layers))
        return False
    else:
        return True

def timeLapse(startTime):
    return (cv2.getTickCount()-startTime)/cv2.getTickFrequency() 

def release(out,cap,client):
    if out is not None:
        out.release()
    cap.release()
    cv2.destroyAllWindows()
    client.disconnect()
    return 

def preprocessed_input(network,frame):
    shape = network.get_input_shape()
    p_frame = cv2.resize(frame, (shape[3], shape[2]))
    p_frame = p_frame.transpose((2,0,1))
    p_frame = p_frame.reshape(1, *p_frame.shape)
    return p_frame

def getCODEC():
    if sys.platform == "linux" or sys.platform == "linux2":
        CODEC = 0x00000021
    elif sys.platform == "darwin":
        CODEC = cv2.VideoWriter_fourcc('M','J','P','G')
    else:
        print("Unsupported OS.")
        exit(1)
    return CODEC

def getSrcDim(cap):
    return int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) , int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
  

def drawBBoxes(frame, result, threshold, width, height):
    count = 0
   
    for box in result[0][0]: # Output shape is 1x1x100x7
        
        conf = box[2]
        if conf >= threshold:
            log.info('Detecting: person with probability: %02.2f',   conf)
            xmin = int(box[3] * width)
            ymin = int(box[4] * height)
            xmax = int(box[5] * width)
            ymax = int(box[6] * height)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 255), 1)
            count = count+1
    return frame, count


     


 
