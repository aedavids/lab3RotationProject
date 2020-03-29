'''
Created on Feb 13, 2020

@author: Andrew Davidson aedavids@ucsc.edu
'''
import logging
from   lowRankMatrixFactorization.lowRankMatrixFactorization import _costMatlab, _XGradient, _ThetaGradient,\
            _pack, _unpack, _cost, _gradient, LowRankMatrixFactorization
import numpy as np
from   scipy import optimize
from   setupLogging import setupLogging
import unittest


################################################################################ 
class TestLowRankMatrixFactorization(unittest.TestCase):
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
    def createTestData(self):
        Y = np.array( [[1,  2,  3,  4],
                       [5,  6,  7,  8],
                       [9, 10, 11, 12]])
        
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
    def testCostEqNoRegulaton(self):
        self.logger.info("BEGIN")
        
        X, Theta, Y, R, regulationRate, numRows, numCols, numLearnedFeatures = self.createTestData()
        J = _costMatlab(X, Theta, Y, R, regulationRate)
        self.logger.info("J: {}".format(J))
        
        expectedJ = 193 # matlab value
        self.logger.warning("possible bug we are calculating mean squared error why are we using 0.5 instead of 1/m or 1/2m")
                    
        self.assertEqual(J, expectedJ)
 
        self.logger.info("END\n")
        
    ################################################################################ 
    def testCostEqWithRegulaton(self):
        self.logger.info("BEGIN")
        
        X, Theta, Y, R, regulationRate, numRows, numCols, numLearnedFeatures = self.createTestData()
        regulationRate = 1
        
        J = _costMatlab(X, Theta, Y, R, regulationRate)
        self.logger.info("J: {}".format(J))
        
        expectedJ = 200 # matlab value
        self.logger.warning("possible bug we are calculating mean squared error why are we using 0.5 instead of 1/m or 1/2m")
                    
        self.assertEqual(J, expectedJ)
 
        self.logger.info("END\n")        
        
    ################################################################################ 
    def test_XGradient(self):
        self.logger.info("BEGIN")
        
        X, Theta, Y, R, regulationRate, numRows, numCols, numLearnedFeatures = self.createTestData()
        result = _XGradient(X, Theta, Y, R, regulationRate)
        self.logger.info("result:\n{}".format(result))
        
        # matlab value
        expectedXGrad =np.array([[ -2, -2],
                                 [-18, -18],
                                 [-34, -34]])

        self.assertTrue( (result == expectedXGrad).all()) 
 
        self.logger.info("END\n")
                
                
    ################################################################################ 
    def test_ThetaGradient(self):
        self.logger.info("BEGIN")
        
        X, Theta, Y, R, regulationRate, numRows, numCols, numLearnedFeatures = self.createTestData()
        result = _ThetaGradient(X, Theta, Y, R, regulationRate)
        self.logger.info("result:\n{}".format(result))
        
        # matlab value
        expectedThetaGrad = np.array([
                                    [ -9,  -9],
                                    [-12, -12],
                                    [-15, -15],
                                    [-18, -18]
                                    ])
        self.assertTrue( (result == expectedThetaGrad).all() ) 
 
        self.logger.info("END\n")       
        
    ################################################################################ 
    def testPackUnpack(self):
        self.logger.info("BEGIN")
        
        X, Theta, Y, R, regulationRate, numRows, numCols, numLearnedFeatures = self.createTestData()
        params, args = _pack(X, Theta, Y, R, regulationRate, numRows, numCols, numLearnedFeatures)
        self.logger.info("params:\n{}".format(params))
        self.logger.info("args:\n{}".format(args))
        self.logger.info("len(args):\n{}".format(len(args)))
        self.logger.info("args[0]:\n{}".format(args[0]))

        
#         X, Theta,              Y,      R,       regulationRate,       numRows,     numCols, numLearnedFeatures
        XResult, ThetaResult, YResult, RResult, regulationRateResult, numRowsResult, numColsResult, numLearnedFeaturesResult = _unpack(params, args)
                    
        self.assertTrue((X == XResult).all())
        self.assertTrue((Theta == ThetaResult).all())
        self.assertTrue((Y == YResult).all())
        self.assertTrue((R == RResult).all())
        self.assertEqual(regulationRate, regulationRateResult)                
        self.assertEqual(numRows, numRowsResult)
        self.assertEqual(numCols, numColsResult)
        self.assertEqual(numLearnedFeatures, numLearnedFeaturesResult)
 
        self.logger.info("END\n")       

    ################################################################################ 
    def testOptimize(self):
        self.logger.info("BEGIN")
        
        X0, Theta0, Y, R, regulationRate, numRows, numCols, numLearnedFeatures = self.createTestData() 
        regulationRate = 0.25 # match matlab
        params, args = _pack(X0, Theta0, Y, R, regulationRate, numRows, numCols, numLearnedFeatures)
        
        # https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.optimize.fmin_cg.html
        display=1 #If True, return a convergence message, followed by xopt.
        fullOutput=True
        retall=False # if true last return argument will be allvecs:l ist of arrays, containing the results at each iteration
        # warnflag = 0 ; success
        # warnflag = 1; The maximum number of iterations was exceeded
        # warnflag = 2;  Gradient and/or function calls were not changing; may mean that precision was lost, i.e., the routine did not converge.
        # you can add a call back function to print out progress
        xopt, fopt, func_calls, grad_calls, warnflag  = optimize.fmin_cg(_cost, params, \
                                                                         fprime=_gradient, \
                                                                         args=args, disp=display, \
                                                                         full_output=fullOutput)
        
        print()
        self.logger.info("xopt:\n{}".format(xopt))
        self.logger.info("fopt: {}".format(fopt))
        
        sizeX = numRows * numLearnedFeatures
        X = xopt[0:sizeX].reshape(numRows, numLearnedFeatures)
    
        ThetaList = xopt[sizeX:]
        Theta =  ThetaList.reshape(numCols, numLearnedFeatures)        
    
        print()
        predictions  = np.matmul(X, Theta.transpose())
        self.logger.info("predictions:\n{}".format(predictions))
        
        #expectedPredictions these values are close to what our matlab test case produced
        expectedPredictions = np.array([[ 2.11223371,  2.43889059,  2.76554747,  3.09220435],
                                        [ 5.25525986,  6.06798565,  6.88071144,  7.69343722],
                                        [ 8.39828602,  9.69708071, 10.99587541, 12.2946701 ]])        
        self.logger.info("predictions - expected:\n{}".format(predictions - expectedPredictions))
        self.assertTrue( np.allclose(predictions, expectedPredictions))
        #self.assertTrue( (predictions == expectedPredictions).all() )

        print()
        error = Y - predictions
        self.logger.info("\nY - prediction:\n{}".format(error))
        print()

        self.logger.info("END\n")         
        
    ################################################################################ 
    def testOptimize2(self):
        self.logger.info("BEGIN")
        
        X0, Theta0, Y, R, regulationRate, numRows, numCols, numLearnedFeatures = self.createTestData() 
        regulationRate = 0.25 # match matlab
        params, args = _pack(X0, Theta0, Y, R, regulationRate, numRows, numCols, numLearnedFeatures)
        
        
        opts = {'maxiter' : None,    # default value.
        'disp' : True,    # non-default value.
        'gtol' : 1e-5,    # default value.
        'norm' : np.inf,  # default value.
        'eps' : 1.4901161193847656e-08}  # default value.
        
        # https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.optimize.minimize.html#scipy.optimize.minimize
        # https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.optimize.OptimizeResult.html#scipy.optimize.OptimizeResult
        # https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.optimize.fmin_cg.html

        optimizeResult = optimize.minimize(_cost, params, jac=_gradient, args=args,
                          method='CG', options=opts)
        
        print()

        sizeX = numRows * numLearnedFeatures
        X = optimizeResult.x[0:sizeX].reshape(numRows, numLearnedFeatures)
        self.logger.info("learned X:\n{}".format(X))
        
        print()
        ThetaList = optimizeResult.x[sizeX:]
        Theta =  ThetaList.reshape(numCols, numLearnedFeatures)        
        self.logger.info("learned Theta:\n{}".format(Theta))
        
        print()
        self.logger.info("success:{}".format(optimizeResult.success))
        self.logger.info("status:{} message:{}".format(optimizeResult.status, optimizeResult.message))
        self.logger.info("final cost: {}".format(optimizeResult.fun))
        self.logger.info("number of iterations: {}".format(optimizeResult.nit))
        self.logger.info(" Number of evaluations of the objective functions : {}".format(optimizeResult.nfev))
        self.logger.info(" Number of evaluations of the objective functions and of its gradient : {}".format(optimizeResult.njev))

        print()
        predictions  = np.matmul(X, Theta.transpose())
        self.logger.info("predictions:\n{}".format(predictions))
        
        #expectedPredictions these values are close to what our matlab test case produced
        expectedPredictions = np.array([[ 2.11223371,  2.43889059,  2.76554747,  3.09220435],
                                        [ 5.25525986,  6.06798565,  6.88071144,  7.69343722],
                                        [ 8.39828602,  9.69708071, 10.99587541, 12.2946701 ]])        
        self.logger.info("predictions - expected:\n{}".format(predictions - expectedPredictions))
        self.assertTrue( np.allclose(predictions, expectedPredictions))
        #self.assertTrue( (predictions == expectedPredictions).all() )

        print()
        error = Y - predictions
        self.logger.info("\nY - prediction:\n{}".format(error))
        print()

        self.logger.info("END\n")                 
                 
                 
    ################################################################################ 
    def testFit(self):
        self.logger.info("BEGIN")
        X0, Theta0, Y, R, regulationRate, numRows, numCols, numLearnedFeatures = self.createTestData() 
        regulationRate = 0.25 # match matlab   
        
#         theMeaningOfLife = 42
#         #np.random.seed(theMeaningOfLife)
#         np.random.default_rng(theMeaningOfLife)

        lrmf = LowRankMatrixFactorization(Y, numLearnedFeatures, regulationRate=regulationRate)  
        X, Theta, optimizeResult = lrmf.fit()   
        
        print()
        self.logger.info("success:{}".format(optimizeResult.success))
        self.logger.info("status:{} message:{}".format(optimizeResult.status, optimizeResult.message))
        self.logger.info("final cost: {}".format(optimizeResult.fun))
        self.logger.info("number of iterations: {}".format(optimizeResult.nit))
        self.logger.info(" Number of evaluations of the objective functions : {}".format(optimizeResult.nfev))
        self.logger.info(" Number of evaluations of the objective functions and of its gradient : {}".format(optimizeResult.njev))

        print()        
        predictions  = np.matmul(X, Theta.transpose())
        print()
        self.logger.info("predictions:\n{}".format(predictions))
        
        expectedPredictions = np.array([        
                                        [ 1.44864811,  2.01587045,  2.85148888,  3.63093097],
                                        [ 5.031414,    5.644386,    6.84986616,  7.83633369],
                                        [ 8.90912557,  9.39212074, 10.77349125, 11.75644623]])

        print()     
        self.logger.info("predictions - expected:\n{}".format(predictions - expectedPredictions))
        self.assertTrue( np.allclose(predictions, expectedPredictions))
        
        
        self.logger.info("END\n")
        
                         
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main() 
