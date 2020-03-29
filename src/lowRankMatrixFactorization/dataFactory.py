'''
Created on Mar 7, 2020

@author: andrewdavidson aedavids@ucsc
'''
from   enum import Enum
import logging
import numpy as np
import os as os
from   pathlib import Path

class RDataSet(Enum):
    TRUTH    = "RTruth"
    TRAIN    = "RTrain"
    VALIDATE = "RValidate"
    TEST     = "RTest"
        
class DataFactory(object):
    '''
    class RDataSet
        enumeration used to specify train, validate, or test
        
    public functions:
        __init__(self, dirPath, prefix=None, suffix=None)
        
        loadR(self, RDateSetType)
        loadTheta(self)
        loadX(self)
        loadY(self)
        
        saveR(self, RDateSetType, R )
        saveTheta(self, theta)
        saveX(self, X)
        saveY(self, Y)
        
        split(self, R, holdOutPercent=0.10)
    '''

    logger = logging.getLogger(__name__)

    ###############################################################################
    def __init__(self, dataDir, prefix=None, suffix=None):
        '''
        Constructor
        '''
        self.dirPath = dataDir
        self.prefix = prefix
        self.suffix = suffix
    
    ###############################################################################
    def loadR(self, RDateSetType):
        fileName = self._fileName("R" + RDateSetType.name)
        try:
            ret = np.loadtxt(fileName, delimiter=',', dtype='bool' )
        except IOError as error:
            self.logger.error("fileName:() error:{}".format(error))
            ret = None
        
        return ret
    
    ###############################################################################
    def loadTheta(self):
        ret = self._loadFloat("Theta")
        return ret

    ###############################################################################
    def loadX(self):
        ret = self._loadFloat("X")
        return ret
    
    ###############################################################################
    def loadY(self):
        ret = self._loadFloat("Y")
        return ret
    
    ###############################################################################
    def saveR(self, RDateSetType, R ):
        self._createOutputDir()
        fileName = self._fileName("R" + RDateSetType.name)
        if R is not None :
            np.savetxt(fileName, R, delimiter=',', fmt="%d") # use %d for boolean

    ###############################################################################
    def saveTheta(self, theta):
        self._createOutputDir()        
        self._saveFloat(theta, "Theta")
        
    ###############################################################################
    def saveX(self, X):
        self._createOutputDir()        
        self._saveFloat(X, "X")
        
    ###############################################################################
    def saveY(self, Y):
        self._createOutputDir()        
        self._saveFloat(Y, "Y")        
    #
    ###############################################################################
    def split(self, R, holdOutPercent=0.10) :
        """
        enable simple sanity test of trained model
        
        splits R into two data sets, a training data set with roughly 1-holdOutPercent of the 
        observations the remaining samples in a separate hold out set. The shape of Y and R remain 
        unchanged
        
        Note split is based on observed values only. For example if holdOutPercent = 50%
        half of the observed values should be each of the returns knockout numpy arrays
        
        input:            
            R
                a logical 2d numpy array. values == True if value was observed
            
            holdOutPercent: default 0.10
                        
        returns (RTrain, RTest)
            RTrain and RTest have the same shape as the R input argument.
            RTrain is the knock out filter for selecting the values of Y to in training set
            RTest is the knock out filter for selecting the values of Y in the test set
        """
        
        #
        # TODO: AEDWIP: FIXME: this is slow
        # its good enough for now. 
        # faster impl
        # 1. get indexes for observed values
        # 2. shuffle list of indexes 
        # 3. split list
        #
        
        # psudo code
        # get index for values in R that have been observed
        # oIdx =  np.argwhere(aR == 1)
        # oIdx == [ [[0,6], [984,234]] ...] you will need to check the spahe
        # calculate number of observations you want in the split
        # generate a random list of int
        # use fancy indexing to select the first split idx
        # split1Idx = oIdx [ random(n) ]
        # use fancy index to select the first split
        # split1 = R[slit1Idx]
        # use knock out to select the second split
        # split2 = np.multiple(R, split1)
        #
        # to test for data leaks
        # np.mulply(split1, split2) == 0
        
        rows, cols = R.shape 
        #n = rows * cols # we should use sum. we only care about observed values
        n = np.sum(R)  
        
        # calculate the size of the test set.
        numKnockOuts = np.floor(n * holdOutPercent).astype('int')
        if numKnockOuts == 0.:
            self.logger.warning("AEDWIP: TODO: holdOutPercent = 0 is not implemented")
        self.logger.debug("np.sum(R):{} numKockOuts:{}".format(n, numKnockOuts))
        RTrain = np.array(R, copy=True)
        RTest = np.zeros(R.shape)
        
        count = 1
        done = False
        miss = 1
        while not done:
            randRow = np.random.randint(low=0, high=rows - 1)
            randCol = np.random.randint(low=0, high=cols - 1)
            self.logger.debug("randRow:{} randCol:{}".format(randRow, randCol))
            if RTrain[randRow, randCol] :
                RTest[randRow, randCol] = True
                RTrain[randRow, randCol] = False
                count += 1
            else:
                miss += 1
                
            if count % 100000 == 0 :
                self.logger.debug("count:{}".format(count))
                count += 1
                
            if miss % 100000 == 0:
                self.logger.debug("miss:{}".format(miss))
                miss += 1
            
            if count >= numKnockOuts:
                done = True
                
        return (RTrain, RTest)
    
    #    
    # private
    # 
    
    ###############################################################################
    def _createOutputDir(self):
        try:
            os.makedirs(self.dirPath,  exist_ok=True)
        except OSError as error: 
            self.logger.error("can not create: {self.resultsDirectory} e:{}".format(self.resultsDirectory, error))
            raise error       
         
    ###############################################################################
    def _fileName(self, name):
        fileName = self.prefix + "_" +  name + "_" + self.suffix + ".csv"
        ret = Path(self.dirPath) / fileName
        
        return ret
        
    ###############################################################################
    def _loadFloat(self, name):
        fileName = self._fileName(name)
        try:
            ret = np.loadtxt(fileName, delimiter=',')
        except IOError as error:
            self.logger.error("file:{} error:{}".format(fileName, error))
            ret = None
            
        return ret    
        
    ###############################################################################
    def _saveFloat(self, obj, name):
        fileName = self._fileName(name)
        np.savetxt(fileName, obj, delimiter=',')

