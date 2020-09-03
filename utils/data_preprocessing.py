import csv
import pandas as pd
import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from onehotencoder import DataFrameEncoder
def data_combine(df):
    df = df.merge(drug_df, how="left", on="pert_id")
    col = df.pop("ECFP")
    df.insert(1, col.name, col)
    df = df.drop(columns=['chemical', 'pert_id','sig_id',  'pert_type', 'pert_idose'])
    # df['vector_gene'] = df[df.columns[5:]].apply(
    #     lambda x: ','.join(x.dropna().astype(str)),
    #     axis=1
    # )
    # return df[["ECFP","cell_id","vector_gene"]]
    return df
def replace_all(df):
    column_index = []
    for i in range(1024):
        column_index.append(i)
    df = df.merge(cell_line_df,how="left",on ="cell_id")
    df[column_index]=df.ECFP.apply(
        lambda x: pd.Series(int(d) for d in str(x)))

    return df.drop(["cell_id","ECFP"],axis=1).dropna(axis=0)

drug_file = pd.read_csv("data\ECFP.csv",
                        names=["pert_id", "chemical", "ECFP"])
cell_line = pd.read_csv("data\ccle.csv")
train_file = pd.read_csv("data\signature_train.csv")
test_file = pd.read_csv("data\signature_test.csv")
dev_file = pd.read_csv("data\signature_dev.csv")

drug_df= pd.DataFrame(drug_file)
cell_line_df = pd.DataFrame(cell_line)
train_df = pd.DataFrame(train_file)
test_df = pd.DataFrame(test_file)
dev_df = pd.DataFrame(dev_file)

train=data_combine(train_df)
test=data_combine(test_df)
dev=data_combine(dev_df)

train=replace_all(train)
test=replace_all(test)
dev=replace_all(dev)

# dev.to_csv(r'D:\Capstone\DeepCE-master\DeepCE-master\DeepCE\data\dev.csv')
# train.to_csv(r'D:\Capstone\DeepCE-master\DeepCE-master\DeepCE\data\train.csv')
# test.to_csv(r'D:\Capstone\DeepCE-master\DeepCE-master\DeepCE\data\test.csv')

train_X,train_y=train[train.columns[978:]],train[train.columns[:978]]
test_X,test_y=test[test.columns[978:]],test[test.columns[:978]]
dev_X,dev_y=dev[dev.columns[978:]],dev[dev.columns[:978]]


print(len(dev_y),len(test_X),len(train_X))
print(len(train_df),len(dev_df),len(test_df))



