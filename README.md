# stocktwits
Scoop Recommendation using Deep Learning 

This Project was developed using PyCharm IDE and Anaconda Python 3 Interpreter.

Required Libraries

1. gensim - https://github.com/RaRe-Technologies/gensim.git
2. Spacy - https://github.com/explosion/spaCy.git
3. pytorch - https://github.com/pytorch/pytorch.git
4. Keras - https://github.com/keras-team/keras.git
5. LightFM - https://github.com/lyst/lightfm.git
6. Tensor Flow  
7. tffm - https://github.com/geffy/tffm.git

Execution steps:

1) In data_prep.py, give the location of Stocktwits files and select the date limit for which you need the training and testing data. 
Remove the bots and split data into 60%, 20%, 20% for train, test and validation files. 

2) After generating train, test and validation files, to get output with predictions, from command line run the following command
libFM -train pathtothefile/train.libfm -validation pathtothefile/validation.libfm -test pathtothefile/test.libfm -task r -dim '1,1,4' -iter 1000 -method mcmc -out pathtothefile/out.libfm
The output file can be used to calculate RMSE of different factorization machines in fm_models.py.

3) The output generated from the parser, can be used as input to baselines. Perform Collaborative and hybrid filtering on the data to calculate
MAP@3 and MRR of different baseline models. After the evaluation, compare the values to decide the best model. This is followed by the 
comparison of MAP@3 and MRR of sequence models in spotlight_sequence.py.

4) The data from the parser is also fed into classification algorithm, It calculates the accuracy and loss for both training and testing data.
By analyzing tags in the tweet body, classification.py algorithm classifies the tweets into their possible target areas. After the 
classification, it generates graphs for model's accuracy and loss.
