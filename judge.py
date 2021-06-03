from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
class my_judge():
    def __init__(self):
        self.df = None
        self.train_score = None
        self.test_score = None
        self.cols = None
        self.X_train = None
        self.Y_train = None
        self.X_test = None
        self.Y_test = None
        self.LR = None

    def create_dataset(self,ranklist, dataset, split = 0.8):
        '''
        Copy the dataset
        Multiply the dataset by its score in the ranklist
        Perform train-test split
        :param ranklist: the sorted rank with features
        :param dataset: the dataset needs to be transformed
        :return: transformed dataset
        '''
        self.df = dataset.copy()
        self.cols = self.df.columns

        for rank in ranklist:
            self.df[rank[1]] *= rank[0]
        
        X = self.df[self.cols[2:]]
        Y = self.df[self.cols[1]]
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(X, Y, test_size = 1-split, random_state=42)
        print(self.X_train.shape, self.Y_train.shape)
        print(self.X_test.shape, self.Y_test.shape)
        return self.df
    
    def fitmodel(self):
        '''
        Minic the fit function of sklearn
        '''
        self.model = LogisticRegression(random_state=1).fit(self.X_train,self.Y_train)

    def report(self):
        '''
        Print the Precision-Recall-F1 table
        Return the accuracy of train and test
        '''
        print(classification_report(self.Y_test, self.model.predict(self.X_test)))
        return self.model.score(self.X_train,self.Y_train), self.model.score(self.X_test, self.Y_test)
    
if __name__ == "__main__":
    print("Additional")