import time

class frmAnalyisSM(object):
    def __init__(self): 
        self.durationStats()

    def durationStats(self):
        self.timer = time.time()
       
    def getPersonEntrytime(self): 
        return self.timer