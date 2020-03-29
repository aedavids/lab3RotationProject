'''
Created on Feb 22, 2020

@author: andrewdavidson aedavids@ucsc.edu
'''

from   DEMETER2.dataFactory import DataFactory as DEMETER2DataFactory
import logging
from   lowRankMatrixFactorization.lowRankMatrixFactorization import LowRankMatrixFactorization
from lowRankMatrixFactorization.dataFactory import DataFactory as LRMFDataFactory
from lowRankMatrixFactorization.dataFactory import RDataSet 
import os as os
from   pathlib import Path

###############################################################################
class LowRankMatrixFactorizationEasyOfUse(object):
    '''
    TODO: 
    
    public functions
        __init__(self, dataDir, fileName, numFeatures, geneFilterPercent, holdOutPercent, 
                 randomize=False, tag=None)
                 
        dipslayOptimizedResults(self, optimizeResult)
                 
        runTrainingPipeLine(self)
        loadAll(self)
        saveAll(self, dataDict)
    '''
    
    logger = logging.getLogger(__name__)

    ###############################################################################
    def __init__(self, dataDir, fileName, numFeatures, geneFilterPercent, holdOutPercent, 
                 randomize=False, tag=None):
        '''
        Constructor
        '''
        
        self.dataDir = dataDir
        self.fileName = fileName
        self.numFeatures = numFeatures
        self.geneFilterPercent = geneFilterPercent
        self.holdOutPercent = holdOutPercent
        
        self.prefix = fileName.split(".")[0] # file should end in something like .tsv. this is more robust
        
        fmt = "n_{}_geneFilterPercent_{}_holdOutPercent_{}"
        self.suffix  = fmt.format(numFeatures, geneFilterPercent, holdOutPercent)
        
        self.randomize = randomize
        self.tag = tag
        
        if self.tag != None:
            self.suffix  += tag
            
        self.resultsDirectory = Path(self.dataDir) / self.suffix
        self.demeter2DataFactory = DEMETER2DataFactory(self.dataDir, self.resultsDirectory, self.fileName, self.suffix)
        self.lrmfDataFactory = LRMFDataFactory( self.resultsDirectory, self.prefix, self.suffix)
        
        
    ###############################################################################
    def dipslayOptimizedResults(self, optimizeResult):
        """
        prints() optimizeResult: an object of type scipy.optimize.OptimizeResult 
        """
        print("success:{}".format(optimizeResult.success))
        print("status:{} message:{}".format(optimizeResult.status, optimizeResult.message))
        print("final cost: {}".format(optimizeResult.fun))
        print("number of iterations: {}".format(optimizeResult.nit))
        print(" Number of evaluations of the objective functions : {}".format(optimizeResult.nfev))
        print(" Number of evaluations of the objective functions and of its gradient : {}".format(optimizeResult.njev))                
            
    ###############################################################################
    def runTrainingPipeLine(self):
        '''
        1) loads and clean the DEMETER2 data, applies geneFilterPercent
        
        2)  splits R so that you can select a train, validation and test set
        The holdOutPercent is used to split R into a train and hold out set.
        The validation and test set are created by spliting the hold out set in half
        
        3) the Low rank matrix factorization model is fit on the DEMETER2 training data
        
        TODO: replace dictionary with a class
        returns: { "DEMETER2" : (Y, R, cellLineNames, geneNames),
                "LowRankMatrixFactorizationModel" : (X, Theta, optimizeResult),
                "filters" : (RTrain, RValidation, RTest))
            }
        
            where the values 
                DEMETER2: the clean numpy version of the raw data
                LowRankMatrixFactorizationModel : The trained, fitted model
                filters: the R knock out matrix filters. Use these to partition Y into test sets
        '''
        self.logger.info("begin load and clean")
        Y, R, cellLineNames, geneNames = self.demeter2DataFactory.loadAndClean(self.geneFilterPercent)
        self.logger.info("load and clean completed")
        
        if self.randomize:
            Y, R = self.demeter2DataFactory.randomizeData(Y, R)
        
        self.logger.info("begin data set split")
        RTrain, RHoldOut = self.lrmfDataFactory.split(R, self.holdOutPercent)
        
        RValidation, RTest = self.lrmfDataFactory.split(RHoldOut, holdOutPercent=0.5)
        self.logger.info("end data set split")
        
        self.logger.info("begin model fit")
        X, Theta, optimizeResult = self._fit(Y, RTrain)
        self.logger.info("end model.fit")
        
        ret = { "DEMETER2" : (Y, R, cellLineNames, geneNames),
                "LowRankMatrixFactorizationModel" : (X, Theta, optimizeResult),
                "filters" : (RTrain, RValidation, RTest)
            }
        return ret
        
    ###############################################################################
    def loadAll(self):
        '''
        returns clean, tidy version of DEMETER2, the trained model, and the knockout filters
        used to select the train, validation, and test sets 
        '''
        cellLineNames, geneNames = self.demeter2DataFactory.loadName()
        
        Theta = self.lrmfDataFactory.loadTheta()
        X = self.lrmfDataFactory.loadX()        
        Y = self.lrmfDataFactory.loadY()
        
        R          =  self.lrmfDataFactory.loadR(RDataSet.TRUTH)
        RTrain      =  self.lrmfDataFactory.loadR(RDataSet.TRAIN)
        RValidation = self.lrmfDataFactory.loadR(RDataSet.VALIDATE)
        RTest       = self.lrmfDataFactory.loadR(RDataSet.TEST)
        
        optimizeResult = None
        
        ret = { "DEMETER2" : (Y, R, cellLineNames, geneNames),
                "LowRankMatrixFactorizationModel" : (X, Theta, optimizeResult),
                "filters" : (RTrain, RValidation, RTest)
            }
        
        return ret                
    ###############################################################################
    def saveAll(self, dataDict):
        '''
        input:
            dataDict see runTrainingPipeLine return values
            
        return:
            path to directory the data was sorted in
        '''
        try:
            os.makedirs(self.resultsDirectory,  exist_ok=True) 
            
            Y, RTruth, cellLinesList, geneNamesList =  dataDict["DEMETER2"]
            
            self.demeter2DataFactory.saveNames(cellLinesList, geneNamesList)
            
            self.lrmfDataFactory.saveR(RDataSet.TRUTH, RTruth)
            
            RTrain, RVal, RTest = dataDict["filters"]
            self.lrmfDataFactory.saveR(RDataSet.TRAIN, RTrain)
            self.lrmfDataFactory.saveR(RDataSet.VALIDATE, RVal)
            self.lrmfDataFactory.saveR(RDataSet.TEST, RTest)
            
            X, Theta, optimizeResult = dataDict["LowRankMatrixFactorizationModel"]
            self.lrmfDataFactory.saveTheta(Theta)
            self.lrmfDataFactory.saveX(X)
            self.lrmfDataFactory.saveY(Y)
            
        except OSError as error: 
            self.logger.error("can not create: {self.resultsDirectory} e:{}".format(self.resultsDirectory, error))
            raise error
        
        return self.resultsDirectory
        
    #
    # private 
    #
    
    ###############################################################################
    def _fit(self, Y, R):
        """
        implements Low Rank Matrix Factorization
        
        returns: (X, Theta, optimizeResult)
            X: final learned features matrix. shape (Y.shape[0], numLearnedFeatures)
            Theta: final learned features matrix. shape (Y.shape[1], numLearnedFeatures)
            optimizeResult: an object of type scipy.optimize.OptimizeResult 
        """
        
        try:
            lmrf = LowRankMatrixFactorization(Y, self.numFeatures, R)
            X, Theta, optimizeResult = lmrf.fit()
        except Exception as e:
            self.logger.error("training error:{}".format(e))
            return
        
        self.logger.debug("END\n")
        return (X, Theta, optimizeResult)        
