import time

class durationSM(object):
    def __init__(self): 
        self.durationStats()

    def durationStats(self):
        self.timer = time.time()
       
    def getEntrytime(self): 
        return self.timer