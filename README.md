
# A Novel Approach to Identifying and Predicting Cancer Vulnerabilities
winter qtr 2020
author: Andrew E. Davidson, aedavids@ucsc.edu
Mentor: Alana Weinstein
Speical thanks to Professor Stuart for his guidance and direction. 


My final lab rotation was in the [Stuart lab at the Univ. of California Santa Cruz](https://sysbiowiki.soe.ucsc.edu/)

To get a better understanding of the this project see presentations/lab3RotationLightningTalk.pptx and presentations/status-2020-03-13.pptx

This project is implemented as collection of python3 packages and jupyter notebooks. You can view the juypter notebooks with out having to start your own juypter server by viewing the notebook source file on github.

## Installation
set up conda env with required packages
```
$ conda create --name labRotation3TensorFlow --file requirements.txt
```

## Starting notebooks

```
cd ~workSpace/UCSC/labRotation3DEMETER2
conda activate labRotation3TensorFlow
export PYTHONPATH="${PYTHONPATH}:`pwd`/src"
jupyter notebook
```

## Running Unit test
```
cd ~workSpace/UCSC/labRotation3DEMETER2
conda activate labRotation3labRotation3TensorFlow
export PYTHONPATH="${PYTHONPATH}:`pwd`/src"
cd src/test
python -m unittest discover .
```

## Creating the prerequisite tidy data sets.
In ./data you will find the raw gene dependency data from project achilles and a coorespond RNA Seq data file. The many of the notebooks assume that you have created a clean tidy data set and trained a low rank matrix factorization model locally. In the future the raw data file should be removed and replace with code to download the files as needed.

Start by uncompressing the zip files in data/

To create the expected LMRF model. Run the trainRandomHoldOut.ipynb juypter notebook. The notebook will train a model with 19 learned features. This was found to be best model. The results will be sorted at data/n_19_geneFilterPercent_0.25_holdOutPercent_0.4. The results include the trained model and a tidy version of the raw data.

To create the auto encoder data set, run the createAutoEncoderDataSet.ipynb juypter notebook. This notebook will create a clean tidy data set. The result will be found in data/autoEncoder

## Table of Contents: Notebook overviews
- requirements.txt
  * list of required python packages
  
### Low Rank Matrix Factorization Model notebooks

- explore.ipynb
  * turns out 14% of D2_Achilles_gene_dep_scores_5by5.tsv are missing
  * we should remove DEMETER2.dataFactory._clean(filter=0.01) and impute any remaining missing values

- createUnitTestData: used to create the 5x5 TSV file src/test/data/D2_Achilles_gene_dep_scores_5by5.tsv 
  
- trainRandomHoldOut:
  * used to create trained data sets of various sizes
  * has some wall clock stats for various hold out sizes and n learned features
  * the results can be found data/ and have names like  holdOut_0.1_numFeature_100
    + notebook saves data to data/ we have to manually move data ot a sub directory to make sure we do not accidently over write it.
    + file examples.
      - Y, RTest, and RTrain have same shape
      - RTrain and RTest are knockout,filter,logical arrays
      - use RTest to select values in Y that are part of the hold out set
      ```
      $ ls data/holdOut_0.1_numFeature_100/
      D2_Achilles_gene_dep_scores_RTest_numFeatures_100.csv 
      D2_Achilles_gene_dep_scores_Theta_numFeatures_100.csv
      D2_Achilles_gene_dep_scores_RTrain_numFeatures_100.csv 
      D2_Achilles_gene_dep_scores_X_numFeatures_100.csv
      ```
      
  * 19 learned features has best performance.

- evaluateRandomHoldOut.ipynb
  * used for low rank matrix hyper parameter tunning

- findSimilarGened.ipynb
- findSimilarCellLines.ipynb
  
### RNA Seq to Gene Dep mapping model
- createAutoEncoderDataSet.ipynb
  * creates data set for rnaSeq2GeneDependency.ipynb
  * data saved to data/autoencoder/
- geneDependencyAutoEncoder
  * use this notebook to debug deep models
- rnaSeq2GeneDependency.ipynb
  * a deep model that maps RNA seq data to gene dependency data


## TODO 
- see notes on improving accuracy on p. 82 , 3/11,  BME notebook # 1

- finish geneDependencyAutoEncoder.ipynb and rnaSeq2GeneDependency.ipynb

- see presentation/status-*.pptx

- summary
  * which RNA seq data should we run
    + CCLEv1_hugo_log2tpm_58581genes_2019-04-15.tsv is probably better than RNAseq_lRPKM_data.csv
    + treehouseID_to_CCLEID.tsv is probably better than  CCLEv1_hugo_log2tpm_58581genes_2019-04-15.tsv
      - uses different gene is. see  sample_info.csv

- need to add documentation to data factory about no need to shuffle Y. we are learn independent regression models for each row and col
    
### low priority TODO
- datafactory _split() is slow it is good enough. see comments in code for faster impl
  - double check no over lap between spliits. test 
    * train,hold = split, val, 
    * test = split(holdOut)
- go through code, remove set random seed. let user of class set that do not default


