'''
Created on Feb 17, 2020

@author: andrewdavidson aedavids@ucsc.edu
'''

import logging
import os as os
import numpy as np
from   pathlib import Path

###############################################################################
class DataFactory(object):
    '''
    loads and stores DEMETER2 data files and Low Rank Matrix Factorization models
    
    public functions:
        __init__(self, dataRootDir, outputDir, fileName, suffix=None)
        loadAndClean(self, rawDataFile, geneFilter=None)
        
        utility functions
        loadName(self)
        randomizeData(self, Y, R)
    '''
    
    logger = logging.getLogger(__name__)

    ###############################################################################
    def __init__(self, dataRootDir, outputDir, fileName, suffix=None):
        '''
        Constructor
            input:
                dataRootDir: the root of all raw data and trained model data 
                example: ./data
                
            outputDir:
                location to save data to
            fileName:
                the name of a DEMETER2 tab separated file
                example: D2_Achilles_gene_dep_scores.tsv
                
        '''
        self.dataRootDir = dataRootDir
        self.outputDir = outputDir
        self.rawDataFile = fileName
        self.prefix = fileName.split(".")[0] # file should end in something like .tsv. this is more robust
        
        if suffix is None:
            suffix = ""
        self.suffix = suffix
            
    ###############################################################################
    def loadAndClean(self, geneFilter=None):
        '''        
        'NA' are converted to 66666666.66666 , R is a logical matrix you can use to filter out these values
        
        input 
            rawDataFile:
                a DEMETER2 tab delimited data set
                example: "data/D2_Achilles_gene_dep_scores.tsv"
            
            shuffle:
                boolean value. If true data randomly orders the cell lines and genes
            
            geneFilter:
                a percentage. For example geneFilter=0.1
                remove all genes if percent of 'NA' is greater than this value
                The over all number of rows will be reduced
                
                default value: None. that is to say do not filter
            
        returns:
                (Y, R, cellLineList, geneNameList)
        '''
        self.logger.debug("BEGIN")
        strMatrix = self._loadRaw()
                
        try:
            Y, R, cellLineNames, geneNames = self._cleanData(strMatrix, geneFilter)
            self.logger.debug("_cleanData completed")
        except Exception as e:
            self.logger.error("cleaning error:{}".format(e))
            return
        
        self.logger.debug("END\n")        
        return (Y, R, cellLineNames, geneNames)
    
    ###############################################################################
    def randomizeData(self, Y, R):
        '''
        create a base line data set to train on. Models trained on true data must perform 
        better than this
        
        arguments:
            Y : 
                A clean Tidy matrix. All values are numeric
            R:
                the corresponding knockout matrix
        
        return:
            (Y, R)
             a copy of Y and R where the columns of each row are randomly rearranged. 
             
        example:
        
            Y = np.array([
                [  1,   2,   3,  4,   5],
                [ 10,  20,  30,  40,  50],
                [100, 200, 300, 400, 500]
                ])
            
            clearly R is not a valid knock out matrix however it makes it easier to understand 
            what the function does
            
            R = np.array([
                [ 'a',  'b',  'c',  'd',  'e'],
                [ 'f',  'g',  'h',  'i',  'j'],
                [ 'k',  'l',  'm',  'n',  'o']
                ])
                
            YOut=  [[  2   5   3   1   4]
                    [ 40  20  30  10  50]
                    [200 100 400 500 300]]
                    
            ROut = [['b' 'e' 'c' 'a' 'd']
                    ['i' 'g' 'h' 'f' 'j']
                    ['l' 'k' 'n' 'o' 'm']]
        '''
        retY = np.copy(Y)
        retR = np.copy(R)
        
        numRows, numCols = Y.shape        
        for r in range(numRows):
            randomRowIdx = np.random.permutation(numCols)
            retY[r,:] = retY[r,randomRowIdx]
            retR[r,:] = retR[r,randomRowIdx]
            
        return (retY, retR)
    
    ###############################################################################
    def loadName(self):
        '''
        returns:
            (list of cell Line Names, list of gene names)
        '''
        clFileName = self._fileName("cellLineNames")
        retCL = np.loadtxt(clFileName, delimiter=',', dtype='str'  )
        
        gnFileName = self._fileName("geneNames")
        retGN = np.loadtxt(gnFileName, delimiter=',', dtype='str'  )
        
        return(retCL, retGN)
    
    ###############################################################################
    def saveNames(self, cellLinesList, geneNamesList):
        '''
        TODO
        '''
        
        self._createOutputDir()
        clFileName = self._fileName("cellLineNames")
        np.savetxt(clFileName, cellLinesList, delimiter=',', fmt="%s")
        
        gnFileName = self._fileName("geneNames")
        np.savetxt(gnFileName, geneNamesList, delimiter=',', fmt="%s")
            
    
    #
    # private
    #
    
    ###############################################################################
    def _createOutputDir(self):
        try:
            os.makedirs(self.outputDir,  exist_ok=True)
        except OSError as error: 
            self.logger.error("can not create: {self.resultsDirectory} e:{}".format(self.resultsDirectory, error))
            raise error

    ###############################################################################
    def _fileName(self, name):
        self.logger.debug("prefix:{} name:{} suffix:{}".format(self.prefix, name, self.suffix))
        fileName = self.prefix + "_" +  name + "_" + self.suffix + ".csv"
        ret = Path(self.outputDir) / fileName
        
        return ret
    
    ###############################################################################    
    def _cleanData(self, strMatrix, geneFilter=None):
        """
        argumens:
            strMatrix:
                a 2d numpy matrix of strings.
                
            geneFilter:
                a percentage. For example geneFilter=0.1
                remove all genes if percent of 'NA' is greater than this value
                The over all number of rows will be reduced
        
        returns:
            (Y, R, cellLineList, geneNameList)
            
            Y:
                2d numpy matrix of floats. missing values have been replaced by np.NAN
            
            R:
                a 2d numpy knockout matrix of booleans with the same shape as Y
                if string value at position was 'NA' boolean value == False
            
            cellLineList:
                list of strings in the first row of strMatrix
            
            geneNameList:
                list of strings in the first col of strMatrix
        """
        self.logger.debug("BEGIN")

        cellLineList = strMatrix[0,1:]
        geneNameList = strMatrix[1:,0]
        
        # remove cell line and gene name data 
        YStr = strMatrix[1:,1:]
                
    
        # fix NA value
        tokenStr = '66666666.66666'
        tokenFloat = 66666666.66666 # np.NAN
        YStr[ YStr == 'NA' ] = tokenStr

        Y = YStr.astype(float)
    
        # construct R
        R = Y != tokenFloat
        
        if geneFilter is not None:
            dropLogical = self._findGenesWithLotsOfMissingData(R, geneFilter)
            selectLocical = dropLogical == False
            Y = Y[selectLocical,:]        
            R = R[selectLocical,:]        
            geneNameList = geneNameList[selectLocical]
            
        self.logger.debug("END\n")
        return (Y, R, cellLineList, geneNameList)
        
    ###############################################################################    
    def _findGenesWithLotsOfMissingData(self, R, percent):
        """
        input:
            missing: a logical matrix. one means value is missing, else zero
            
            precent: identify genes that are missing greater than percent values
            
        returns:
            logical vector.
        """
        missing = R == False
        rows, cols = missing.shape
        #print(missing.shape)
        
        rowAxis = 1
        missingSums = np.sum(missing, axis=rowAxis)
        self.logger.debug(missingSums.shape)
        
        percentMissing = missingSums / (1. * cols) # make sure denominator is a real num
        idx = np.argsort(percentMissing)[-50:]
        self.logger.debug(idx)
        self.logger.debug(missingSums[idx])
        self.logger.debug(percentMissing[idx])
    
        dropLogical = percentMissing > percent
        self.logger.debug("\npercent:{} numToDrop{}".format(percent, np.sum(dropLogical))) 
        self.logger.debug(missingSums[dropLogical])
        
        return dropLogical
          
    ###############################################################################
    def _loadRaw(self):
        self.logger.debug("BEGIN")
        # TODO: AEDWIP: clean up
        dataFile = Path(self.dataRootDir) / self.rawDataFile
        try:
            ret = np.loadtxt(dataFile, dtype=str, delimiter="\t")
            self.logger.debug("np.loadTxt completed")
            
        except Exception as e:
            self.logger.error("np.loadTxt error:{}".format(e))
            ret = None
            
        self.logger.debug("END\n")
        return ret
