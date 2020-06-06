import os
import logging as log

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
     


 
