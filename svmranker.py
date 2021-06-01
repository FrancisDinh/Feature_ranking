import numpy as np 
import pandas as pd 
from keras.utils import np_utils
import eli5
from eli5.sklearn import PermutationImportance
from sklearn import svm
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

class my_svm_ranker():
    '''
    This is my sample production code for my ranker
    '''
    def __init__(self, datapath):
        self.datapath = datapath
        #self.model = RandomForestRegressor()
        self.model = svm.SVC(kernel='linear', C=0.025)
        self.X_train = None
        self.Y_train = None
        self.X_test = None
        self.Y_test = None
        self.flag = False
        self.cols = None
        self.X_label = None
        self.Y_label = None
    
    
    def preprocess(self):
        '''
        Practically, it exits, however in this situation, it doesnt
        since data needs not to be.
        '''
        pass


    def dataLoader(self,X_col,Y_col,split=1.0):
        '''
        Create features and labels
        Plit train test sets

        :param X_col: column numbers of X
        :param Y_col: column number of Y
        :param split: train test split coefficient 
        :exception: return -1
        '''
        if not os.path.exists(self.datapath):
            self.flag = True
            return -1


        data = pd.read_csv(self.datapath)
        self.cols = data.columns
        self.X_label = self.cols[X_col]
        self.Y_label = self.cols[Y_col]
        X = data[self.X_label]
        Y = data[self.Y_label]

        if split == 1:
            self.X_train, self.Y_train = X, Y
            self.X_test, self.Y_test = pd.DataFrame([]),pd.DataFrame([])
        else:
            self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(X, Y, test_size=1-split, random_state=42)
        
        # split_range = int(len(data)*split)
        # self.X_train, self.Y_train = X.iloc[:split_range], Y.iloc[:split_range]
        # self.X_test, self.Y_test = X.iloc[split_range:], Y.iloc[split_range:]
        print(self.X_train.shape, self.Y_train.shape)
        print(self.X_test.shape, self.Y_test.shape)

    def train(self):
        '''
        Minic the fit function of built-in model

        :exception: return -1
        '''
        if self.flag:
            return -1
        self.model.fit(self.X_train, self.Y_train)
        
     
    def permutation(self, n = 20):
        '''
        This function evaluate the role of each feature to re-check the output of the model when we fit. And since we assume that the output of the model is correct, we will need to prove it by Permutation Importance
        '''
        if len(self.X_test) == 0 or len(self.Y_test) == 0:
            perm = PermutationImportance(self.model, n_iter = n).fit(self.X_train, self.Y_train)
        else:
            perm = PermutationImportance(self.model, n_iter = n).fit(self.X_test, self.Y_test)
        print(perm.feature_importances_)
        return eli5.show_weights(perm, feature_names = list(self.X_label))                    
        

    def report(self, top = 11, plot = True):
        '''
        Report the ranking list
        Show the bar plot

        :param top: number of top elements, default = 11
        :param plot: show bar plot, default True
        :return rank: ranking 
        :exception: return -1
        '''
        if self.flag:
            return -1
        # Get coefficience as the metrics to evaluate ranking
        rank = sorted(zip(map(lambda x: round(x,3), 
                      self.model.coef_[0]), self.X_label), reverse=True)
        if plot:
            importance = list(zip(*rank))[0][:top]
            sensor = list(zip(*rank))[1][:top]

            fig = plt.figure(figsize = (16,8))
            sensor = list(zip(*rank))[1][:top]
            value = list(zip(*rank))[0][:top]
            plt.bar(sensor, value, align="center")
            plt.ylabel("Importance")
            for idx, val in enumerate(value):
                if val > 0:
                    plt.text(x = idx - 0.125 , y = val + 0.01, s = str(val) , fontdict = dict(fontsize=12))
                else:
                    plt.text(x = idx - 0.125 , y = 0.025, s = str(val) , fontdict = dict(fontsize=12))
            plt.tight_layout()
            plt.show()
        return rank

if __name__ == "main":
    model = my_ranker("../../task_data.csv")
    model.dataLoader([2,3,4,5,6,7,8,9,10,11],1,0.8)
    model.train()
    model.report()
    model.permutation()
