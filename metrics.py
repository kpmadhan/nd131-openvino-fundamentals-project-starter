import os
import logging as log
import sys


def format_bytes(size):
    # 2**10 = 1024
    power = 2**10
    n = 0
    power_labels = {0 : '', 1: 'k', 2: 'm', 3: 'g'}
    while size > power:
        size /= power
        n += 1
    return round(size,2), power_labels[n]+'b'

def getSize(filename):
    st = os.stat(filename)
    s,u = format_bytes(st.st_size)
    return '{0} {1}'.format(s,u) 


def getErrorPercent(observed , metric):
    if metric == "people" :
        actual = 6
    elif metric == "duration":
        actual = 18  
    return  '{0} %'.format(str(round((((observed-actual)/actual) * 100),2)))  #Manually observed the video and got the actual metrics to calculate the error percentage
  