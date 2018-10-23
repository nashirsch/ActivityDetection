'''
Created by:

@author: Andrea F Daniele - TTIC - Toyota Technological Institute at Chicago
May 14, 2017 - Chicago - IL

'''

import sys
import numpy as np

class ProgressBar():

    def __init__(self, maxVal=100, precision=4, doneMessage=True ):
        self.maxVal = float(maxVal)
        self.doneMessage = doneMessage
        self.precision = precision
        self.currentLength = -1
        self.currentVal = 0.0
        self.barParts = [ '[0%' ]
        for i in range(10,101,10): self.barParts.extend( ['.'] * self.precision + ['%d%%' % i] )
        self.barParts[-1] += ']'
        if doneMessage: self.barParts[-1] += ' Done!'
        self.barLength = len(self.barParts)
        self.step = float(self.barLength-1) / self.maxVal

    def next(self):
        newLength = int(np.floor( (self.currentVal + 1.0) * self.step ))
        if newLength > self.currentLength and newLength <= self.barLength:
            for i in range(self.currentLength+1, newLength+1):
                sys.stdout.write(self.barParts[i]); sys.stdout.flush()
            if newLength == self.barLength-1: print
            self.currentLength = newLength
        self.currentVal += 1

    def setMessage(self, message):
        self.barParts[-1] = '100%%] :: %s\n' % message
