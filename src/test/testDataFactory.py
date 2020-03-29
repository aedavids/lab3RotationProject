'''
Created on Feb 22, 2020

@author: andrewdavidson aedavids@ucsc.edu
'''

# from DEMETER2.dataFactory import DataFactory
from   DEMETER2.dataFactory import DataFactory as DEMETER2DataFactory
from lowRankMatrixFactorization.dataFactory import DataFactory as LRMFDataFactory
#from DEMETER2.modelLowRankMatrixFactorization import ModelLowRankMatrixFactorization
from DEMETER2.lowRankMatrixFactorizationEasyOfUse import LowRankMatrixFactorizationEasyOfUse as LrmfEoU

import logging
import numpy as np
from   pathlib import Path
from   setupLogging import setupLogging
import shutil

import unittest

################################################################################ 
class TestDataFactory(unittest.TestCase):
    configFilePath = setupLogging( default_path='logging.test.ini.json')
    logger = logging.getLogger(__name__)
    logger.info("using logging configuration file:{}".format(configFilePath))   

    ################################################################################ 
    def setUp(self):
        pass

    ################################################################################ 
    def tearDown(self):
        pass

    ################################################################################ 
    def testName(self):
        pass
    
    ################################################################################ 
    def createTestData(self):
        Y = np.array( [[1,  2,  3,  4],
                       [5,  6,  7,  8],
                       [9, 10, 11, 12],
                       
                       [1,  2,  3,  4],
                       [5,  6,  7,  8],
                       [9, 10, 11, 12],
                       
                       [1,  2,  3,  4],
                       [5,  6,  7,  8],
                       [9, 10, 11, 12]
                       ])
        
        numRows, numCols = Y.shape
        numLearnedFeatures = 2
        X = np.ones( (numRows, numLearnedFeatures) )
        Theta = np.ones( (numCols, numLearnedFeatures) )
        R = np.ones(Y.shape)        
        regulationRate = 0.0
        
        return((X, Theta, Y, R, regulationRate, numRows, numCols, numLearnedFeatures))    

    ################################################################################ 
    def testTemplate(self):
        self.logger.info("BEGIN")
 
        self.logger.info("END\n")

    ################################################################################ 
    def testSplit(self):
        self.logger.info("BEGIN")
        
        AEDWIPSeed = 42 # should we be using np.random.randomstate.uniform()
        np.random.seed(AEDWIPSeed)
        lrmfDataFactory = LRMFDataFactory(dataDir=None, prefix=None, suffix=None)
        
        X, Theta, Y, R, regulationRate, numRows, numCols, numLearnedFeatures = self.createTestData()
        
        RTrain, RTest = lrmfDataFactory.split(R, holdOutPercent = 0.20)
        self.logger.info("np.sum(R):{}".format(np.sum(R)))
        self.logger.info("RTrain.shape:{} RTest.shape{}".format(RTrain.shape, RTest.shape))
        self.logger.info("np.sum(RTrain):{} RTrain:\n{}".format(np.sum(RTrain), RTrain))
        self.logger.info("np.sum(RTest):{}RTest:\n{}".format(np.sum(RTest), RTest))
        
        expectedRTrain = np.array(
                           [[1., 1., 1., 1.],
                            [1., 1., 0., 1.],
                            [0., 1., 1., 1.],
                            
                            [1., 1., 1., 1.],
                            [1., 1., 0., 1.],
                            [1., 1., 1., 1.],
                            
                            [0., 1., 0., 1.],
                            [0., 1., 1., 1.],
                            [1., 1., 1., 1.]])
        
        self.assertEqual(np.sum(R), np.sum(RTrain) + np.sum(RTest))
        self.assertTrue((RTrain == expectedRTrain).all())

        
        expectedRTest = np.array(
                           [[0., 0., 0., 0.],
                            [0., 0., 1., 0.],
                            [1., 0., 0., 0.],
                            
                            [0., 0., 0., 0.],
                            [0., 0., 1., 0.],
                            [0., 0., 0., 0.],
                            
                            [1., 0., 1., 0.],
                            [1., 0., 0., 0.],
                            [0., 0., 0., 0.]])
        
        self.assertTrue((RTest == expectedRTest).all())
        
        self.assertTrue( (R == RTrain + RTest).all())
 
        self.logger.info("END\n")
        
    ################################################################################ 
    def testSplit50Percent(self):
        self.logger.info("BEGIN")
        
        AEDWIPSeed = 42 # should we be using np.random.randomstate.uniform()
        np.random.seed(AEDWIPSeed)
        lrmfDataFactory = LRMFDataFactory(dataDir=None, prefix=None, suffix=None)

        X, Theta, Y, R, regulationRate, numRows, numCols, numLearnedFeatures = self.createTestData()
        self.logger.info("R:\n{}".format(R))
        RTrain, RTest = lrmfDataFactory.split(R, holdOutPercent = 0.50)
        self.logger.info("np.sum(R):{}".format(np.sum(R)))
        self.logger.info("RTrain.shape:{} RTest.shape{}".format(RTrain.shape, RTest.shape))
        self.logger.info("np.sum(RTrain):{} RTrain:\n{}".format(np.sum(RTrain), RTrain))
        self.logger.info("np.sum(RTest):{}RTest:\n{}".format(np.sum(RTest), RTest))
        
        self.assertTrue((RTrain != RTest).all())
        
        expectedRTrain = np.array([  [1., 0., 0., 1.],
                                     [1., 0., 0., 1.],
                                     [0., 1., 0., 1.],
                                     [0., 0., 0., 1.],
                                     [0., 1., 0., 1.],
                                     [0., 0., 1., 1.],
                                     [0., 0., 0., 1.],
                                     [0., 1., 1., 1.],
                                     [1., 1., 1., 1.]])
        
        self.assertEqual(np.sum(R), np.sum(RTrain) + np.sum(RTest))
        self.assertTrue((RTrain == expectedRTrain).all())
        
        self.logger.info("np.sum(RTrain == RTest):{}".format(np.sum(RTrain == RTest)))
        self.assertEqual(np.sum(RTrain == RTest), 0)
        self.logger.info("np.sum(RTrain != RTest):{}".format(np.sum(RTrain != RTest)))
        self.assertEqual(np.sum(RTrain != RTest), 36)

        expectedRTest = np.array([   [0., 1., 1., 0.],
                                     [0., 1., 1., 0.],
                                     [1., 0., 1., 0.],
                                     [1., 1., 1., 0.],
                                     [1., 0., 1., 0.],
                                     [1., 1., 0., 0.],
                                     [1., 1., 1., 0.],
                                     [1., 0., 0., 0.],
                                     [0., 0., 0., 0.]])
        
        self.assertTrue((RTest == expectedRTest).all())
        self.assertTrue( (R == RTrain + RTest).all())
        
        cellLineNames = ["c1", "c2"]
        geneNames = ["g1", "g2", "g3"]
        
        dataDir = "data"
        fileName = "D2_Achilles_gene_dep_scores_5by5.tsv"
        numFeatures =  None
        geneFilterPercent = None
        holdOutPercent = None
        easyOfUse = LrmfEoU(dataDir, fileName, numFeatures, geneFilterPercent, holdOutPercent)
        
        RValidation = None
        optimizeResult = None
        saveDict = { "DEMETER2" : (Y, R, cellLineNames, geneNames),
                "LowRankMatrixFactorizationModel" : (X, Theta, optimizeResult),
                "filters" : (RTrain, RValidation, RTest)
            }
        
        easyOfUse.saveAll(saveDict)
        
        retDict = easyOfUse.loadAll()
            
        retRTrain, retRValidation, retRTest = retDict["filters"]
        self.assertTrue((retRTrain == expectedRTrain).all())
        self.assertTrue((retRTest == expectedRTest).all())
        
        self.logger.info("np.sum(RTrainRet == RTestRet):{}".format(np.sum(retRTrain == retRTest)))
        self.assertEqual(np.sum(retRTrain == retRTest), 0)
        
        self.logger.info("np.sum(RTrainRet != RTest):{}".format(np.sum(retRTrain != retRTest)))
        self.assertEqual(np.sum(retRTrain != retRTest), 36) 
        
        self.logger.info("END\n")        
        
    ################################################################################ 
    def testHoldOutSaveAndLoad(self):
        self.logger.info("BEGIN")
        
        # what is the numpy type
        a = np.array(([0.0, 1.0, 2,0]))
        b = np.array(([0.0, 1.0, 3,0]))
        c = a != b
        self.logger.info("type(c):{}".format(type(c)))
        self.logger.info("type(c[0]):{}".format(type(c[0])))

        dataDir = "data"
        fileName = "D2_Achilles_gene_dep_scores_5by5.tsv"
        numFeatures =  2
        geneFilterPercent = 0.222
        holdOutPercent = 0.444
        easyOfUse = LrmfEoU(dataDir, fileName, numFeatures, geneFilterPercent, holdOutPercent)
        
        expectedDict = easyOfUse.runTrainingPipeLine()
        expectedRTrain, expectedRValidation, expectedRTest = expectedDict["filters"]
        
        expectedX, expectedTheta, expectedOpt = expectedDict["LowRankMatrixFactorizationModel"]
        
        expectedY , expectedR , expectedCellLines, expectedGeneNames, = expectedDict["DEMETER2"]
        # being unit test
        
        easyOfUse.saveAll(expectedDict)

        retDict = easyOfUse.loadAll()
        RTrainRet, RValidationRet, RTestRet = retDict["filters"]
        
        self.logger.info("expectedRTrain:\n{}".format(expectedRTrain))
        self.logger.info("RTrainRet:\n{}".format(RTrainRet))
        self.logger.info("RValidationRet:\n{}".format(RValidationRet))
        self.logger.info("RTestRet:\n{}".format(RTestRet))
        
        self.assertTrue(  (expectedRTrain == RTrainRet).all() )
        self.assertTrue( (expectedRValidation == RValidationRet).all() )
        self.assertTrue(  (expectedRTest == RTestRet).all() )        
        
        retX, retTheta, retOpt = retDict["LowRankMatrixFactorizationModel"]
        self.assertTrue( (expectedTheta == retTheta).all() ) 
        
        retY , retR , retCellLines, retGeneNames, = expectedDict["DEMETER2"]
        self.assertTrue( (expectedCellLines == retCellLines).all )
 
        self.logger.info("END\n")
        
        
    ################################################################################ 
    def testTrainAll(self):
        """
        train on entire data set
        """
        self.logger.info("BEGIN")
        
        # remove data from previous test runs
        try:
            shutil.rmtree("./data/n_2_geneFilterPercent_0_holdOutPercent_0.1")
        except FileNotFoundError:
            pass
        
        
        # train a model and save to disk
        numFeatures = 2
        dataDir = "data"
        dataFile = "D2_Achilles_gene_dep_scores_5by5.tsv"

        geneFilterPercent = 0
        holdOutPercent = 0.1
        easyOfUse = LrmfEoU(dataDir, dataFile, numFeatures, geneFilterPercent, holdOutPercent)
        
        expectedDict = easyOfUse.runTrainingPipeLine()
        expectedRTrain, expectedRValidation, expectedRTest = expectedDict["filters"]
        expectedX, expectedTheta, expectedOpt = expectedDict["LowRankMatrixFactorizationModel"]
        expectedY , expectedR , expectedCellLines, expectedGeneNames, = expectedDict["DEMETER2"]
        
        
        easyOfUse.saveAll(expectedDict)

        retDict = easyOfUse.loadAll()
        RTrainRet, RValidationRet, RTestRet = retDict["filters"]
        retX, retTheta, retOpt = retDict["LowRankMatrixFactorizationModel"]
        retY , retR , retCellLines, retGeneNames, = expectedDict["DEMETER2"]

        # load model from disk
        self.logger.info("expectedR:\n{}".format(expectedR))
        self.logger.info("retR:\n{}".format(retR))
        
        self.assertTrue( (expectedR==retR).all() )
        self.assertTrue( (expectedTheta==retTheta).all()) 
        self.assertTrue( (expectedCellLines == retCellLines).all )        

        self.logger.info("END\n")
        
    ################################################################################ 
    def testCleanNoPreFilter(self):
        '''
        no pre filtering step to remove genes with lots of missing value
        '''
        self.logger.info("BEGIN")
        dataFactory = DEMETER2DataFactory(dataRootDir="data", 
                                          outputDir=None, \
                                          fileName="D2_Achilles_gene_dep_scores_5by5.tsv")
                                          
        strMatrix = dataFactory._loadRaw()
        Y, R, cellLineNames, geneNames = dataFactory._cleanData(strMatrix)

        self.logger.info("Y:\n{}".format(Y))
        expectedY = np.array(
                [[ 6.66666667e+07, -1.15242032e-01, -2.31724480e-02, -2.33374641e-02],
                [ 8.41725922e-02,  6.66666667e+07, -1.54187815e-01, -7.90055311e-02],
                [ 2.07020442e-01,  1.07433673e-02, -7.21022146e-02,  4.56106276e-02],
                [ 6.21916667e-02, -4.98086050e-02,  2.21367779e-02,  6.17088692e-02]])
        
        self.assertTrue(np.isclose(Y, expectedY).all())
        
        self.logger.info("R:\n{}".format(R))
        expectedR = np.array( [[False,  True,  True,  True],
                             [ True, False,  True,  True],
                             [ True,  True,  True,  True],
                             [ True,  True,  True,  True]])
        self.assertTrue(np.isclose(R, expectedR).all())

        self.logger.info("cellLineNames:{}".format(cellLineNames))
        expectedCellLineNames = np.array(['"143B_BONE"', '"22RV1_PROSTATE"', '"2313287_STOMACH"',\
                                          '"697_HAEMATOPOIETIC_AND_LYMPHOID_TISSUE"'])
        self.assertTrue((cellLineNames == expectedCellLineNames).all())
        
        self.logger.info("geneNames:{}".format(geneNames))
        expectedGeneNames = np.array(['"A1BG (1)"', '"NAT2 (10)"', '"ADA (100)"' ,'"CDH2 (1000)"'])
        self.assertTrue((geneNames == expectedGeneNames).all())

        self.logger.info("END\n")        
        
    ################################################################################ 
    def testCleanWithPreFilter(self):
        '''
        pre filtering step to remove genes with lots of missing value
        '''
        self.logger.info("BEGIN")
        numFeatures = 2
        dataFactory = DEMETER2DataFactory(dataRootDir="data",
                                          outputDir=None, \
                                          fileName="D2_Achilles_gene_dep_scores_5by5.tsv")

        strMatrix = dataFactory._loadRaw()
        Y, R, cellLineNames, geneNames = dataFactory._cleanData(strMatrix, geneFilter=0.1)

        self.logger.info("Y:\n{}".format(Y))
        expectedY = np.array(
                [[ 2.07020442e-01,  1.07433673e-02, -7.21022146e-02,  4.56106276e-02],
                [ 6.21916667e-02, -4.98086050e-02,  2.21367779e-02,  6.17088692e-02]])
        
        self.assertTrue(np.isclose(Y, expectedY).all())
        
        self.logger.info("R:\n{}".format(R))
        expectedR = np.array([[ True,  True,  True,  True],
                             [ True,  True,  True,  True]])
        self.assertTrue(np.isclose(R, expectedR).all())

        self.logger.info("cellLineNames:{}".format(cellLineNames))
        expectedCellLineNames = np.array(['"143B_BONE"', '"22RV1_PROSTATE"', '"2313287_STOMACH"',\
                                          '"697_HAEMATOPOIETIC_AND_LYMPHOID_TISSUE"'])
        self.assertTrue((cellLineNames == expectedCellLineNames).all())
        
        self.logger.info("geneNames:{}".format(geneNames))
        #expectedGeneNames = np.array(['"A1BG (1)"', '"NAT2 (10)"', '"ADA (100)"' ,'"CDH2 (1000)"'])
        expectedGeneNames = np.array(                              ['"ADA (100)"', '"CDH2 (1000)"'])
        self.assertTrue((geneNames == expectedGeneNames).all())

        self.logger.info("END\n")                
        
        
    ################################################################################ 
    def testPathLib(self):
        '''
        test to see how to use Path. should make it easy do build paths
        '''
        
        self.logger.info("BEGIN")
        
        pathNoSlash = "a/b/c"
        p = Path(pathNoSlash)
        self.logger.info(p)
        
        self.logger.info( p / "d/e")
        
        pathWithSlash = "a/b/c/"
        p = Path(pathWithSlash)
        self.logger.info(p)
        
        self.logger.info( p / "d/e")
        
        self.logger.info("END\n")        
        
    ################################################################################ 
    def testRandomizeData(self):
        self.logger.info("BEGIN")
        np.random.seed(42)
        
        Y = np.array([
                [  1,   2,   3,  4,   5],
                [ 10,  20,  30,  40,  50],
                [100, 200, 300, 400, 500]
                ])
        
        R = np.array([
                [ 'a',  'b',  'c',  'd',  'e'],
                [ 'f',  'g',  'h',  'i',  'j'],
                [ 'k',  'l',  'm',  'n',  'o']
                ])  
        
        dataFactory = DEMETER2DataFactory(dataRootDir="data", 
                                          outputDir=None, \
                                          fileName="D2_Achilles_gene_dep_scores_5by5.tsv")        
        resultY, resultR = dataFactory.randomizeData(Y, R)
        
        self.logger.info("resultY:\n{}".format(resultY))
        self.logger.info("resulR:\n{}".format(resultR))
        
        expectedY=  np.array([[  2,   5,   3,   1,   4],
                              [ 40,  20,  30,  10,  50],
                              [200, 100, 400, 500, 300]])
                
        expectedR = np.array([['b', 'e', 'c', 'a', 'd'],
                              ['i', 'g', 'h', 'f', 'j'],
                              ['l', 'k', 'n', 'o', 'm']])
                    
        self.assertTrue((resultY == expectedY).all() )
        self.assertTrue((resultR == expectedR).all() )
              
        self.logger.info("END\n")

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()