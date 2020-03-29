'''
Created on Feb 12, 2020

@author: Andrew Davidson aedavids@ucsc.edu
'''
import logging
import numpy as np
from   scipy import optimize

###############################################################################
class LowRankMatrixFactorization(object):
    '''
    Given a matrix Y , learns a matrix X and Theta such that X * Theta.transpose = Y
    R boolean matrix with the same shape as Y. a value of True means value was observered
    
    Only trains on observerved values
    
    public functions
        __init__(self, Y, n, R=None, regulationRate=1, seed=42)
        fit(self, method='CG', opts={ 'maxiter' : None,    # default value.
                           'disp'    : True,    # non-default value.
                           'gtol'    : 1e-5,    # default value.
                           'norm'    : np.inf,  # default value.
                           'eps'     : 1.4901161193847656e-08 }   # default value.
                        ):
    
    TODO: AEDWIP: 
    make this work like tensorflow fit()
    1. add optional constructor argument RValidation, and verbose
    2. add callback function to optimizer
    3. in call back
        calculate and save cost on training set and cost on validation set
        if verbose print costs
    4. store cost values in objects. 
        user can plot cost by iteration to validation learning 
        we expect the validation cost to be higher than the training cost
    '''
    logger = logging.getLogger(__name__)

    ###############################################################################
    def __init__(self, Y, n, R=None, regulationRate=1, seed=42):
        '''
        arguments:
            Y: a 2 d numpy array
            n: the number of features to learn
            R: provides support for sparse input
               R is a logical matrix used to mark which values in Y are observed
               R(i, j) = 1 if the i-th row was observed by  j-th  column of Y
              else zero 
              
              default assume all values in R are 1
              
            requlationRate:
                lambda. default value is 1
                
            seed:
                random.default_rng(seed)
            
        notes call np.random.seed( your seed ) to get consistent test results
        '''
        s = Y.shape
        self.numRows = s[0]
        self.numCols = s[1]
        
        self.Y = Y
        self.n = n
        if R is None:
            self.R = np.ones(Y.shape)
        else :
            self.R = R
            
        self.regulationRate = regulationRate
        
        gen = np.random.default_rng(seed)
        self.X = gen.uniform(size=( self.numRows, self.n) )
        self.Theta = gen.uniform( size=(self.numCols, self.n) )
                
    ###############################################################################
    def fit(self, method='CG', opts={ 'maxiter' : None,    # default value.
                           'disp'    : True,    # non-default value.
                           'gtol'    : 1e-5,    # default value.
                           'norm'    : np.inf,  # default value.
                           'eps'     : 1.4901161193847656e-08 }   # default value.
                        ):
        '''
        trains a Low Rank Matrix Factorization model.
        
        Note: train can take a long time
        
        arguments:
            method: 
                    gradient based optimization algorithm. default value 'CG' 
                    the conjugate gradient algorithm is based on work of Polak and Ribiere
                    
            opts: 
                    configuration parameters. default values 
                    { 'maxiter' : None,    # default value.
                           'disp'    : True,    # non-default value.
                           'gtol'    : 1e-5,    # default value.
                           'norm'    : np.inf,  # default value.
                           'eps'     : 1.4901161193847656e-08 # default value. 
                    }
                    
        returns:
            (X, Theta, optimizeResult)
                X: final learned features matrix. shape (Y.shape[0], numLearnedFeatures)
                Theta: final learned features matrix. shape (Y.shape[1], numLearnedFeatures)
                optimizeResult: an object of type scipy.optimize.OptimizeResult 
            
        ref:
            https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.optimize.minimize.html#scipy.optimize.minimize
            https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.optimize.OptimizeResult.html#scipy.optimize.OptimizeResult
            https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.optimize.fmin_cg.html
        '''
        
        params, args = _pack(self.X, self.Theta, self.Y, self.R, self.regulationRate, 
                             self.numRows, self.numCols, self.n)
        optimizeResult = optimize.minimize(_cost, params, jac=_gradient, args=args,
                          method=method, options=opts)
        
        sizeX = self.numRows * self.n
        X = optimizeResult.x[0:sizeX].reshape(self.numRows, self.n)
    
        ThetaList = optimizeResult.x[sizeX:]
        Theta =  ThetaList.reshape(self.numCols, self.n)           
        
        return (X, Theta, optimizeResult)


###############################################################################
def _pack(X, Theta, Y, R, regulationRate, numRows, numCols, numLearnedFeatures): 
    '''
    aedwip
    '''       
    args = (Y, R, numLearnedFeatures, regulationRate)
    
    xf = X.flatten() 
    tf = Theta.flatten()
    
    params = np.append( xf, tf )
    
    ret = (params, args)
    return ret

###############################################################################
def _unpack(params, args): 
    '''
    aedwip
    '''       
    Y, R, numLearnedFeatures, regulationRate = args
    numRows, numCols = Y.shape
    
    sizeX = numRows * numLearnedFeatures
    X = params[0:sizeX].reshape(numRows, numLearnedFeatures)
    
    sizeTheta = numCols * numLearnedFeatures
    ThetaList = params[sizeX:]
    Theta =  ThetaList.reshape(numCols, numLearnedFeatures)
    
    return (X, Theta, Y, R, regulationRate, numRows, numCols, numLearnedFeatures)

###############################################################################
def _cost(params, *args):
    '''
    version of cost function that works with Scipy.optimize
    '''
    argList = [a for a in args]
    X, Theta, Y, R, regulationRate, numRows, numCols, numLearnedFeatures = _unpack(params, argList)
    ret = _costMatlab(X, Theta, Y, R, regulationRate)
    return ret
    
###############################################################################
def _costMatlab(X, Theta, Y, R, regulationRate):
    '''
    returns J the value of the cost function for X, Theta, Y, and R
    
    Scipy optimizer requires cost be implemented as function. Scipy does not support
    class member function
    
    arguments:
        X, Theta, Y, R, requlationRate
    
    commented out code is Matlab version of cost function()
    '''
    
    LowRankMatrixFactorization.logger.debug("X:\n{}".format(X))
    LowRankMatrixFactorization.logger.debug("Theta:\n{}".format(Theta))
    LowRankMatrixFactorization.logger.debug("R:\n{}".format(R))
    
    # commented out code is Matlab version of cost function()
    
    #predictions = (X * Theta');
    predictions = np.matmul( X, Theta.transpose() ) 
    
    #error = predictions - Y;
    error = predictions - Y
    
    #sqError = error .* error;
    sqError = np.multiply(error, error )
    
    # % only calculate a cost value if R(i,j) is 1
    #SumIfR = sqError .* R;
    sumIfR = np.multiply(sqError, R )
      
    #sumSqError = sum(SumIfR(:)); % convert SumIfR into vector then sum
    sumSqError = sumIfR.sum() # convert SumIfR into vector then sum
    
    #J = (1 / 2) * sumSqError;
    J = 0.5 * sumSqError
     
    #%
    #% add regulation terms for theta to the cost
    #%
    #squareTheta = Theta .* Theta;
    squareTheta = np.multiply( Theta, Theta )
    
    #sumSquareTheta = sum(squareTheta(:)); % convert matrix into vector then sum
    sumSquareTheta = np.sum( squareTheta ) # convert matrix into vector then sum
    
    #thetaRegulation = (lambda / 2) * sumSquareTheta;
    thetaRegulation = regulationRate / 2.0 * sumSquareTheta
    
    J = J + thetaRegulation;
     
    #%
    #% add regulation terms for X to the cost
    #%
    #squareX = X .* X;
    squareX = np.multiply( X, X)
    
    #sumSquareX = sum(squareX(:)); % convert matrix into vector then sum
    sumSquareX = np.sum(squareX); # convert matrix into vector then sum
    
    xRegulation = (regulationRate / 2.0) * sumSquareX;
    J = J + xRegulation;
    
    return J
    
###############################################################################
def _gradient(params, *args):
    '''
    # commented out code is Matlab version of cost function()

    '''
    argList = [a for a in args] # AEDWIP clean up hack
    X, Theta, Y, R, regulationRate, numRows, numCols, numLearnedFeatures = _unpack(params, argList)
    XGrad = _XGradient(X, Theta, Y, R, regulationRate)
    YGrad = _ThetaGradient(X, Theta, Y, R, regulationRate)
    
    ret = np.concatenate( (XGrad.flatten(), YGrad.flatten()) )
    return ret

###############################################################################
def _ThetaGradient(X, Theta, Y, R, regulationRate):
    '''
    # commented out code is Matlab version of cost function()
    '''
    # matlab explained
    # X_grad is (rows, learnedFeatures)
    # Theta_grad is (cols, learnedFeatures
    # X_grad(:) is (rows * learnedFeatures, 1)
    # Thet_grad(:) is (cols * learnedFeatures, 1)
    # grad is ( ((rows * learnedFeatures) + (cols * learnedFeatures)), 1 )
    # grad = [X_grad(:); Theta_grad(:)];

    
    #%
    #% calculate partial derivatives for Theta_grad
    #%
    
    ThetaGradRet = np.zeros(Theta.shape)
    numRows, numCols = Y.shape
    numLearnedFeatures = X.shape[1]
    
    #for j = 1 : num_users % for all users 
    for j in range(numCols):
        # % assume we have Movie features 
        #jdx = find(R(:,j)==1); % idx is the list of all movies rated by user j
        jdx = np.argwhere(R[:,j]==1).flatten()

        #tmpX = X(jdx',:); % features for all movies watched by user J
        tmpX = X[jdx.transpose(), :]; #% features for all movies watched by user J
        
        #tmpY = Y(jdx', j); % actual star rating for movies watched by user j
        tmpY = Y[jdx.transpose(), j] # actual star rating for movies watched by user j
        
        #tmpTheta = Theta(j,:); % parameters for user j 
        tmpTheta = Theta[j,:]  #% parameters for user j 
         
        #prediction = tmpX * tmpTheta';
        prediction = np.matmul( tmpX, tmpTheta.transpose() )
        
        #error = prediction - tmpY; % 
        error = prediction - tmpY
       
        #grad = error' * tmpX; 
        grad = np.matmul( error.transpose(), tmpX ) 
         
#         # Theta_grad(j,:) = grad;
        ThetaGradRet[j,:] = grad;
#         
#         #%
#         #% add regulation terms
#         #%
#         #ThetaGradRet(j,:)  = ThetaGradRet(j,:) + lambda * tmpTheta;
#         ThetaGradRet[j,:]  = ThetaGradRet[j,:] + regulationRate * tmpTheta;
    #end %for    
    
    return ThetaGradRet

###############################################################################
def _XGradient(X, Theta, Y, R, regulationRate):
    '''
    # commented out code is Matlab version of cost function()
    '''
    #%
    #% calculate partial derivatives for X_grad
    #%
    
    numRows, numCols = Y.shape
    #numLearnedFeatures = X.shape[1]
    XGradRet = np.zeros(X.shape)
    #for i = 1 : numRows 
    for i in range(numRows) :
        # idx = find(R(i,:)==1); # idx is the list of all user that have rated movie i
        # tmpTheta = Theta(idx,:); % parameters for user i 
        idx = np.argwhere(R[i,:]==1).flatten()
        tmpTheta = Theta[idx, :]
        
        #tmpY = Y(i, idx); % actual star rating for movies watched by user i
        tmpY = Y[i, idx] # actual star rating for movies watched by user i
        
        #features = X(i,:); % features for move i
        features = X[i,:] 
                
        # % error = prediction .- tmpY; # octave version
        #error = (features *tmpTheta' - tmpY);
        error = np.matmul( features,tmpTheta.transpose() ) - tmpY
        
        #X_grad(i,:) =  error * tmpTheta; 
        XGradRet[i,:] = np.matmul(error, tmpTheta)
    
        # %
        # % add regulation term
        # %
        # X_grad(i,:) = X_grad(i,:)  + lambda * features;
        XGradRet[i,:] = XGradRet[i,:] + regulationRate * features
        
    #end %for
    
    return XGradRet