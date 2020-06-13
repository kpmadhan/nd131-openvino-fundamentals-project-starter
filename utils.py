import os
import logging as log
import cv2
import sys
from constants import coco_category_map as COCO_MAP

log.basicConfig(
    level=log.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        log.FileHandler("debug.log"),
        log.StreamHandler()
    ]
)

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

def getSize(filename):
    st = os.stat(filename)
    return st.st_size
  

def drawBBoxes(frame, result, threshold, width, height,prev_frame_count,f_n):
    
    count = 0
    for box in result[0][0]: # Output shape is 1x1x100x7
       
        if int(box[1]) in COCO_MAP:
                class_name =   COCO_MAP.get(int(box[1]))
        else:
                class_name = 'Unknown'

        conf = box[2]

        if conf > 0:
            #log.info('- Detected object {} with probability: {:.2f}'.format(class_name,conf))

            if(class_name == 'person'): #In the current context , our object of interest is person.
                
                log.info('** Detected person with probability: {:.2f}'.format(conf))
               
                xmin = int(box[3] * width)
                ymin = int(box[4] * height)
                xmax = int(box[5] * width)
                ymax = int(box[6] * height)

                if conf >= threshold:

                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 1)
                    cv2.putText(frame, "{} {:.2f}".format(class_name, conf), (xmin+2, ymin+5),cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 1)
                    #cv2.putText(frame, "{} {:.2f}".format(class_name, conf), (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 255), 1, cv2.LINE_AA)
                    count = count+1

                if (prev_frame_count != 0):  # To work around a poorly performing model , an intution that if the person was detected in the earlier frame , they may not disappear in the next frame , and probably the confidence turned low
                    if (prev_frame_count > count and conf >= (0.75 * threshold)): 
                        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 255), 1)
                        cv2.putText(frame, "{} {:.2f}".format(class_name, conf), (xmin+2, ymin+15),cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 255), 1)
                        f_n = f_n+1
                        count = prev_frame_count
                    
                    if (prev_frame_count > count and conf >= (0.50 * threshold)): 
                        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 1)
                        cv2.putText(frame, "{} {:.2f}".format(class_name, conf), (xmin+2, ymin+15),cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)
                        f_n = f_n+1
                        count = prev_frame_count
                    

    return frame, count , f_n


     


 
