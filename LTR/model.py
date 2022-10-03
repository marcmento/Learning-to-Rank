import pandas as pd
import numpy as np
from catboost import CatBoostRanker, Pool
from sklearn.model_selection import cross_val_score
import csv
from sklearn.model_selection import GroupKFold

#Funtion to cleap up the data
def strip_header(the_file):
  with open(the_file, 'r') as f:
    for line in f:
        if line.startswith('#'):
            header = line
        else:
            break
  the_header = header[1:].strip().split('\t')
  df = pd.read_csv(the_file,comment='#',names=the_header,sep='\t')
  return df

#Method to produce final run file
def get_run_file(train_index, vali_index, best_iter, best_lr):
    train_df = dtrain.iloc[train_index,:]
    vali_df = dtrain.iloc[vali_index,:]

    queries_train = train_df['QueryID']
    X_train = train_df.drop(['QueryID', 'Label','Docid'], axis=1)
    y_train = train_df["Label"]

    query_vali = vali_df['QueryID']
    valiq = vali_df['QueryID']
    valid = vali_df['Docid']
    X_vali = vali_df.drop(['QueryID', 'Label','Docid'], axis=1)
    y_vali = vali_df["Label"]

    query_test = dtest['QueryID']
    testq = dtest['QueryID']
    testd = dtest['Docid']
    X_test = dtest.drop(['QueryID', 'Docid'], axis=1)

    train = Pool(
        data=X_train,
        label=y_train,
        group_id=queries_train
    )

    validate = Pool(
        data=X_vali,
        label=y_vali,
        group_id=query_vali
    )

    test = Pool(
        data=X_test,
        group_id=query_test
    )

    model = CatBoostRanker(objective = 'YetiRankPairwise', iterations = best_iter, learning_rate = best_lr)
    model.fit(train, eval_set = validate)
    ypred = model.predict(test)

    with open("A2.tsv", 'w', newline='') as outf:
        i = 0
        while i < len(ypred):
            outf.write(testq.iloc[i] + '\t'+ testd.iloc[i] + '\t'+ str(ypred[i]) + '\n')
            i += 1
    
# Test cross folds
def cross_folds():
    queries_train = dtrain['QueryID']
    X_train = dtrain
    y_train = dtrain["Label"]

    model = CatBoostRanker(objective = 'YetiRankPairwise')
    gkf = GroupKFold(n_splits=5)

    j = 1
    for train_index , vali_index in gkf.split(X_train, y_train, groups = queries_train):
        X_trainCV , X_valiCV = X_train.iloc[train_index,:],X_train.iloc[vali_index,:]
        y_trainCV , y_valiCV = y_train[train_index] , y_train[vali_index]
        queries_trainCV , queries_valiCV = queries_train[train_index] , queries_train[vali_index]
        valiq = X_valiCV['QueryID']
        valid = X_valiCV['Docid']
        
        X_trainCV = X_trainCV.drop(['QueryID', 'Label','Docid'], axis=1)
        X_valiCV = X_valiCV.drop(['QueryID', 'Label','Docid'], axis=1)
        
        train = Pool(
            data = X_trainCV,
            label = y_trainCV,
            group_id = queries_trainCV
        )
        
        validate = Pool(
            data = X_valiCV,
            label = y_valiCV,
            group_id = queries_valiCV
        )

        #COMMENTED OUT TRAINING AND WRITING
        
        # model.fit(train, eval_set = validate)
        # cvPred = model.predict(validate)
        
        # filename = 'CV' + '_' + str(j) + ".run"
        # with open(filename, 'w', newline='') as outf:
        #     i = 0
        #     while i < len(cvPred):
        #         outf.write(valiq.iloc[i] + '\t'+ valid.iloc[i] + '\t'+ str(cvPred[i]) + '\n')
        #         i += 1
                
        # filename = 'CV' + '_' + str(j) + ".qrels"     
        # with open(filename, 'w', newline='') as outf:
        #     i = 0
        #     while i < len(y_valiCV):
        #         outf.write(valiq.iloc[i] + '\t' + '0' + '\t'+ valid.iloc[i] + '\t'+ str(y_valiCV.iloc[i]) + '\n')
        #         i += 1

        if j == 2:
            Best_train_index = train_index
            Best_vali_index = vali_index
        j += 1

    return Best_train_index, Best_vali_index

# Test parameters
def parameter_checks(train_index, vali_index):
    queries_train = dtrain['QueryID']
    X_train = dtrain
    y_train = dtrain["Label"]

    TRAINX , VALIX = X_train.iloc[train_index,:],X_train.iloc[vali_index,:]
    TRAINY , VALIY = y_train[train_index] , y_train[vali_index]
    QTRAIN , QVALI = queries_train[train_index] , queries_train[vali_index]
    valiq = VALIX['QueryID']
    valid = VALIX['Docid']
    
    TRAINX = TRAINX.drop(['QueryID', 'Label','Docid'], axis=1)
    VALIX = VALIX.drop(['QueryID', 'Label','Docid'], axis=1)
    
    train = Pool(
        data = TRAINX,
        label = TRAINY,
        group_id = QTRAIN
    )
    
    validate = Pool(
        data = VALIX,
        label = VALIY,
        group_id = QVALI
    )

    j = 0
    iterations = [1000,2000,3000]
    while j < 3:
        model = CatBoostRanker(objective = 'YetiRankPairwise', iterations = iterations[j])
        # model.fit(train, eval_set = validate)
        # yPred = model.predict(validate)
        
        # filename = 'Iter' + '_' + str(iterations[j]) + ".run"
        # with open(filename, 'w', newline='') as outf:
        #     i = 0
        #     while i < len(yPred):
        #         outf.write(valiq.iloc[i] + '\t'+ valid.iloc[i] + '\t'+ str(yPred[i]) + '\n')
        #         i += 1
                
        # filename = 'Iter' + '_' + str(iterations[j]) + ".qrels"     
        # with open(filename, 'w', newline='') as outf:
        #     i = 0
        #     while i < len(VALIY):
        #         outf.write(valiq.iloc[i] + '\t' + '0' + '\t'+ valid.iloc[i] + '\t'+ str(VALIY.iloc[i]) + '\n')
        #         i += 1

        if j == 0:
            best_iter = iterations[0]
        j += 1

    j = 0
    learning_rate = [0.03 , 0.1]
    while j < 2:
        model = CatBoostRanker(objective = 'YetiRankPairwise', iterations = 1000, learning_rate = learning_rate[j])
        # model.fit(train, eval_set = validate)
        # yPred = model.predict(validate)
        
        # filename = 'LR' + '_' + str(iterations[j]) + ".run"
        # with open(filename, 'w', newline='') as outf:
        #     i = 0
        #     while i < len(yPred):
        #         outf.write(valiq.iloc[i] + '\t'+ valid.iloc[i] + '\t'+ str(yPred[i]) + '\n')
        #         i += 1
                
        # filename = 'LR' + '_' + str(iterations[j]) + ".qrels"     
        # with open(filename, 'w', newline='') as outf:
        #     i = 0
        #     while i < len(VALIY):
        #         outf.write(valiq.iloc[i] + '\t' + '0' + '\t'+ valid.iloc[i] + '\t'+ str(VALIY.iloc[i]) + '\n')
        #         i += 1

        if j == 1:
            best_lr = learning_rate[1]
        j += 1

    return best_iter, best_lr

# Main
if __name__ == "__main__":
    dtrain = strip_header('train.tsv')
    dtrain.groupby('QueryID')

    dtest = strip_header('test.tsv')
    dtest.groupby('QueryID')

    sweep = False

    if sweep == True:
        train_index, vali_index = cross_folds()
        best_iter, best_lr = parameter_checks(train_index,vali_index)
        get_run_file(train_index, vali_index, best_iter, best_lr)
    else:
        train_index, vali_index = cross_folds()
        get_run_file(train_index, vali_index, 1000, 0.1)

