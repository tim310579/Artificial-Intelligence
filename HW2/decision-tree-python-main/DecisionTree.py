# Data wrangling 
import pandas as pd 

# Array math
import numpy as np 

# Quick value count calculator
from collections import Counter

from sklearn.metrics import confusion_matrix


class Node: 
    """
    Class for creating the nodes for a decision tree 
    """
    def __init__(
        self, 
        Y: list,
        X: pd.DataFrame,
        min_samples_split=None,
        max_depth=None,
        depth=None,
        node_type=None,
        rule=None,
        category_num=None
    ):
        # Saving the data to the node 
        self.Y = Y 
        self.X = X

        # Saving the hyper parameters
        self.min_samples_split = min_samples_split if min_samples_split else 2
        self.max_depth = max_depth if max_depth else 20

        # Default current depth of node 
        self.depth = depth if depth else 0

        # Extracting all the features
        self.features = list(self.X.columns)
        
        # Type of node 
        self.node_type = node_type if node_type else 'root'

        # Rule for spliting 
        self.rule = rule if rule else ""

        # Category numbers
        self.category_num = category_num if category_num else 2

        # Calculating the counts of Y in the node 
        self.counts = Counter(Y)
        
        # Getting the GINI impurity based on the Y distribution
        self.gini_impurity = self.get_GINI()

        # Sorting the counts and saving the final prediction of the node 
        counts_sorted = list(sorted(self.counts.items(), key=lambda item: item[1]))
        
        # Getting the last item
        yhat = None
        if len(counts_sorted) > 0:
            yhat = counts_sorted[-1][0]
        
        # Saving to object attribute. This node will predict the class with the most frequent class
        self.yhat = yhat 

        # Saving the number of observations in the node 
        self.n = len(Y)

        # Initiating the left and right nodes as empty nodes
        self.left = None 
        self.right = None 

        # Default values for splits
        self.best_feature = None 
        self.best_value = None

        # Define is or not leaf
        #self.is_leaf = True if len(self.counts) == 1 else False
        #print(self.left, self.right)
        

    @staticmethod
    def GINI_impurity(y_count):
        """
        Given the observations of a binary class calculate the GINI impurity
        """
        n = 0
        # Ensuring the correct types
        for i in range(len(y_count)):
            if y_count[i] is None:
                y_count[i] = 0
            n += y_count[i]

        # Getting the total observations
        #n = y1_count + y2_count
        
        # If n is 0 then we return the lowest possible gini impurity
        if n == 0:
            return 0.0

        # Getting the probability to see each of the classes
        p = []
        for i in range(len(y_count)):
            p.append(y_count[i] / n)
        
        # Calculating GINI
        gini = 1
        for i in range(len(p)):
            gini = gini - (p[i] ** 2)
        #print(gini)
        # Returning the gini impurity
        return gini

    @staticmethod
    def ma(x: np.array, window: int) -> np.array:
        """
        Calculates the moving average of the given list. 
        """
        return np.convolve(x, np.ones(window), 'valid') / window

    def get_GINI(self):
        """
        Function to calculate the GINI impurity of a node 
        """
        y_count = []
        # Getting the 0~n counts
        for i in range(self.category_num):
            y_count.append(self.counts.get(i, 0))
        #y1_count, y2_count = self.counts.get(0, 0), self.counts.get(1, 0)
        #print(y1_count, y2_count)
        # Getting the GINI impurity
        #print(y_count)
        #return self.GINI_impurity(y1_count, y2_count)
        return self.GINI_impurity(y_count)

    def best_split(self) -> tuple:
        """
        Given the X features and Y targets calculates the best split 
        for a decision tree
        """
        # Creating a dataset for spliting
        df = self.X.copy()
        df['Y'] = self.Y

        # Getting the GINI impurity for the base input 
        GINI_base = self.get_GINI()
        
        # Finding which split yields the best GINI gain 
        max_gain = 0

        # Default best feature and split
        best_feature = None
        best_value = None

        for feature in self.features:
            # Droping missing values
            Xdf = df.dropna().sort_values(feature)

            # Sorting the values and getting the rolling average
            xmeans = self.ma(Xdf[feature].unique(), 2)

            for value in xmeans:
                # Spliting the dataset 
                left_counts = Counter(Xdf[Xdf[feature]<value]['Y'])
                right_counts = Counter(Xdf[Xdf[feature]>=value]['Y'])
                
                # Getting the Y distribution from the dicts
                y_left = []
                y_right = []
                for i in range(self.category_num):
                    y_left.append(left_counts.get(i, 0))
                    y_right.append(right_counts.get(i, 0))
                #y0_left, y1_left, y0_right, y1_right = left_counts.get(0, 0), left_counts.get(1, 0), right_counts.get(0, 0), right_counts.get(1, 0)

                # Getting the left and right gini impurities
                gini_left = self.GINI_impurity(y_left)
                gini_right = self.GINI_impurity(y_right)

                # Getting the obs count from the left and the right data splits
                n_left = np.sum(y_left)
                n_right = np.sum(y_right)
                
                # Calculating the weights for each of the nodes
                w_left = n_left / (n_left + n_right)
                w_right = n_right / (n_left + n_right)

                # Calculating the weighted GINI impurity
                wGINI = w_left * gini_left + w_right * gini_right

                # Calculating the GINI gain 
                GINIgain = GINI_base - wGINI
                
                # Checking if this is the best split so far 
                if GINIgain > max_gain:
                    best_feature = feature
                    best_value = value 

                    # Setting the best gain to the current one 
                    max_gain = GINIgain

        return (best_feature, best_value)

    def grow_tree(self):
        """
        Recursive method to create the decision tree
        """
        # Making a df from the data 
        df = self.X.copy()
        df['Y'] = self.Y
        #print(self.depth, self.max_depth)
        #print(self.n, self.min_samples_split)
        # If there is GINI to be gained, we split further 
        if (self.depth < self.max_depth) and (self.n >= self.min_samples_split):

            # Getting the best split 
            best_feature, best_value = self.best_split()
            #print(best_feature, best_value)
            if best_feature is not None:
                # Saving the best split to the current node 
                self.best_feature = best_feature
                self.best_value = best_value

                # Getting the left and right nodes
                left_df, right_df = df[df[best_feature]<=best_value].copy(), df[df[best_feature]>best_value].copy()

                # Creating the left and right nodes
                left = Node(
                    left_df['Y'].values.tolist(), 
                    left_df[self.features], 
                    depth=self.depth + 1, 
                    max_depth=self.max_depth, 
                    min_samples_split=self.min_samples_split, 
                    node_type='left_node',
                    rule=f"{best_feature} <= {round(best_value, 3)}",
                    category_num=self.category_num
                    )

                self.left = left 
                self.left.grow_tree()

                right = Node(
                    right_df['Y'].values.tolist(), 
                    right_df[self.features], 
                    depth=self.depth + 1, 
                    max_depth=self.max_depth, 
                    min_samples_split=self.min_samples_split,
                    node_type='right_node',
                    rule=f"{best_feature} > {round(best_value, 3)}",
                    category_num=self.category_num
                    )

                self.right = right
                self.right.grow_tree()
                

    def print_info(self, width=4):
        """
        Method to print the infromation about the tree
        """
        # Defining the number of spaces 
        const = int(self.depth * width ** 1.5)
        spaces = "-" * const
        
        if self.node_type == 'root':
            print("Root")
        else:
            print(f"|{spaces} Split rule: {self.rule}")
        print(f"{' ' * const}   | GINI impurity of the node: {round(self.gini_impurity, 2)}")
        print(f"{' ' * const}   | Class distribution in the node: {dict(self.counts)}")
        print(f"{' ' * const}   | Predicted class: {self.yhat}")
          

    def print_tree(self):
        """
        Prints the whole tree from the current node to the bottom
        """
        self.print_info() 
        
        if self.left is not None: 
            self.left.print_tree()
        
        if self.right is not None:
            self.right.print_tree()

    def predict(self, X:pd.DataFrame):
        """
        Batch prediction method
        """
        predictions = []

        for _, x in X.iterrows():
            values = {}
            for feature in self.features:
                values.update({feature: x[feature]})
        
            predictions.append(self.predict_obs(values))
        
        return predictions

    def predict_obs(self, values: dict) -> int:
        """
        Method to predict the class given a set of features
        """
        cur_node = self
        #print(cur_node.best_feature)
        while cur_node.depth < cur_node.max_depth:
            if cur_node.left == None and cur_node.right == None: break #is leaf
            # Traversing the nodes all the way to the bottom
            best_feature = cur_node.best_feature
            best_value = cur_node.best_value
            #print(cur_node.best_feature, cur_node.best_value, cur_node.is_leaf)
            if cur_node.n < cur_node.min_samples_split:
                break 

            if (values.get(best_feature) < best_value):
                if self.left is not None:
                    cur_node = cur_node.left
            else:
                if self.right is not None:
                    cur_node = cur_node.right
                    
            
            
        return cur_node.yhat

def plant_a_tree(d, cat_col, d_valid_X):
    d = d.sample(n=len(d)*2,replace=True).reset_index(drop=True)
    category_num = len(d[cat_col].value_counts())
    
    X = d.drop(columns=[cat_col])
    Y = d[cat_col].values.tolist()
    #print(Y)
    # Initiating the Node
    root = Node(Y, X, max_depth=20, min_samples_split=3, category_num=category_num)

    # Getting teh best split
    root.grow_tree()

    d_valid_X_subset1 = d_valid_X.copy()
    d_valid_X_subset1['yhat'] = root.predict(d_valid_X_subset1)

    
    return d_valid_X_subset1['yhat']

def do_train_and_votes(d, train_valid_size, tree_num, category_col):

    d_valid = d[int(len(d)*train_valid_size):len(d)]    #valid data
    d = d[0:int(len(d)*train_valid_size)]   #train data
    
    d_valid_X = d_valid.drop(columns=[category_col])    #to predict, not need category
    d_valid_Y = d_valid[category_col].values.tolist()   #target

    result = pd.DataFrame()
    for i in range(tree_num):
        result = pd.concat([result, plant_a_tree(d, category_col, d_valid_X)], axis = 1)

    result = result.T
    votes = []
    #vote fot highest votes
    for col in result.columns:
        votes.append(result[col].mode()[0])

    real_Y = d_valid_Y
    
    correct = 0
    for i in range(len(votes)):
        if votes[i] == real_Y[i]: correct += 1

    mat = confusion_matrix(real_Y, votes)
    #confusion matrix
    
    print(mat)
    print(correct, len(votes), correct/len(votes))
    
def wdbc(train_valid_size, tree_num):
    d = pd.read_csv("../data/wdbc.data", header=None).dropna()
    #replace some label to number
    d[1].replace(['B'], 0,inplace = True )
    d[1].replace(['M'], 1,inplace = True )
    
    d = d.sample(frac=1).reset_index(drop = True) #shuffle
    d = d.drop(columns=[0]) #useless feature
    do_train_and_votes(d, train_valid_size, tree_num, 1)    #split
    
def wpbc(train_valid_size, tree_num):
    d = pd.read_csv("../data/wpbc.data", header=None).dropna()
    #replace some label to number
    d[1].replace(['N'], 0,inplace = True )
    d[1].replace(['R'], 1,inplace = True )
    for col in d.columns:
        d[col].replace(["?"], int(d[col].mode()[0]), inplace=True)  #delete ? value
        d[col] = d[col].astype('float')
    d = d.sample(frac=1).reset_index(drop = True) #shuffle
    d = d.drop(columns=[0]) #useless feature
    do_train_and_votes(d, train_valid_size, tree_num, 1)    #split
    
def wine(train_valid_size, tree_num):
    d = pd.read_csv("../data/wine.data").dropna()
    #replace some label to number
    d['category'].replace([1], 0,inplace = True )
    d['category'].replace([2], 1,inplace = True )
    d['category'].replace([3], 2,inplace = True )
    
    d = d.sample(frac=1).reset_index(drop = True) #shuffle
    do_train_and_votes(d, train_valid_size, tree_num, 'category')   #split

def iris(train_valid_size, tree_num):
    d = pd.read_csv("../data/iris.data").dropna()
    #replace some label to number
    d['category'].replace(['Iris-setosa'], 0,inplace = True )
    d['category'].replace(['Iris-versicolor'], 1,inplace = True )
    d['category'].replace(['Iris-virginica'], 2,inplace = True )
    
    d = d.sample(frac=1).reset_index(drop = True) #shuffle
    do_train_and_votes(d, train_valid_size, tree_num, 'category')   #split
    
def glass(train_valid_size, tree_num):
    d = pd.read_csv("../data/glass.data", header=None).dropna()
    print(d)
    #replace some label to number
    d[10].replace([1], 0,inplace = True )
    d[10].replace([2], 1,inplace = True )
    d[10].replace([3], 2,inplace = True )
    d[10].replace([4], 3,inplace = True )
    d[10].replace([5], 4,inplace = True )
    d[10].replace([6], 5,inplace = True )
    d[10].replace([7], 6,inplace = True )
    
    d = d.sample(frac=1).reset_index(drop = True) #shuffle
    d = d.drop(columns=[0]) #useless feature(ID)
    do_train_and_votes(d, train_valid_size, tree_num, 10)   #split
    
if __name__ == '__main__':
    # Reading data
    #d = pd.read_csv("../data/iris.data")[['length1', 'length2', 'length3', 'length4', 'category']].dropna()
    #d = pd.read_csv("../data/wine.data")[['a', 'b','c','d','e','f','g','h','i','j','k','l','m', 'category']].dropna()

    #wdbc(train_valid_size = 0.7, tree_num = 1)
    #wpbc(train_valid_size = 0.7, tree_num = 1)
    #wine(train_valid_size = 0.7, tree_num = 1)
    #iris(train_valid_size = 0.7, tree_num = 1)
    glass(train_valid_size = 0.7, tree_num = 1)
    glass(train_valid_size = 0.7, tree_num = 5)
    glass(train_valid_size = 0.7, tree_num = 11)
    glass(train_valid_size = 0.7, tree_num = 15)
    glass(train_valid_size = 0.7, tree_num = 25)
