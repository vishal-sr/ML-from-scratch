#import required libraries
import numpy as np # for matrix operations
import pandas as pd # for data manipulation

class LinearRegression:
    """
    Description: Linear regression is a model which finds the relationship between 
                 dependent variable vs independent variables(one or more) by fitting 
                 line to the data.
    """
    
    def __init__(self, alpha = 0.01, n_iter = 3000):
        
        # alpha -> learning rate.(Initially 0.01, generally - 0.001, 0.01, 0.1, 0.5, 1)
        # n_iter -> total number of gradient descent steps to take. (Initially 3000, minimum - 1000 iterations)
        # weights & bias -> coefficients of each feature (if there is n number of features
        #                                               then n number of weights and one bias will be there).
        #                                               In [y = m*x+c], m -> weights and c -> bias.
        
        self.alpha = alpha
        self.n_iter = n_iter
        self.weights = None
        self.bias = None
    
    def mean_squared_error(self, X, y): # also called cost function
        
        # mean_squared_error (MSE) -> Average squared distance between predicted value and actual value.
        #                            the word average(mean) used because the error is calculated for each 
        #                            training examples and took the average(mean) for all training examples.
        # the term (np.dot(X, self.weights) + self.bias) is the predicted value of every train examples.
        # y - actual value of every training examples.
        
        error = np.sum(((np.dot(X, self.weights) + self.bias) - y) ** 2)
        return error / (2 * float(len(X)))
            
    def fit(self, X, y):
        
        # fitting the training examples.
        
        n_samples, n_features = X.shape # n_samples -> total number of training examples.
                                        # n_features -> total number of features in each training examples.
            
        self.weights = np.zeros(n_features) # initializing weights to n_features number of zeros.
        self.bias = 0 # initializing bias to zero.
        
        for _ in range(self.n_iter): 
            
            # for loop to run gradient descent.
            # each loop means each gradient descent step.
        
            y_predicted = np.dot(X, self.weights) + self.bias # y_predicted -> predicted value of each train examples.
            
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y)) # differentiate cost function with respect to weights.
            db = (1 / n_samples) * np.sum(y_predicted - y) # differentiate cost function with respect to bias.
    
            self.weights -= self.alpha * dw # Updating the weights (taking one step of gradient descent).
            self.bias -= self.alpha * db # Updating the bias (taking one step of gradient descent).
            
    def predict(self, data):
        
        # getting new data and predicting the target variable(dependent variable)
        
        return np.dot(data, self.weights) + self.bias

data = pd.read_csv('50_startups.csv')

X = data.iloc[:, :-1]
y = data.iloc[:, -1].values

def categorical_to_numerical(data):
    columns = data.columns.values # getting the columns of the dataset
    
    for column in columns: # runs the loop for every column of the dataset.
        numerical_value_dict = {} # creating an empty dictionary (initially).
                                  # the dictionary will contain categorical value of 
                                  # a specific column and its numercal value.
        
        def get_numerical_value(categorical_value): # function will get the categorical value and fetch 
                                                    # the numerical value from numerical_value_dict 
                                                    # dictionary and return it.
                    
            return numerical_value_dict[categorical_value]
        
        if data[column].dtype != np.int64 and data[column].dtype != np.float64: 
                                                    # checking whether the column is numerical or not.
                                                    # if the column is not numerical then the following set of code will
                                                    # be executed.
            
            column_content = data[column].values.tolist() # fetching the elements of the column.
            unique_elements = set(column_content) # storing the 'unique' elements of the column.
            
            x = 0
            for unique in unique_elements: # runs for all the 'unique' value of the column.
                if unique not in numerical_value_dict: # checking whether the numerical_value_dict dictionary contain this 
                                                       # 'unique' value, if not then it will add the 'unique' value as key and 
                                                       # 'x' as its value.
                            
                    numerical_value_dict[unique] = x   # adding 'unique' value and 'x'.
                    x += 1 #increamenting 'x' to store next unique value.
            
            data[column] = list(map(get_numerical_value, data[column])) # mapping for all element of all categorical columns.
            
    return data

X = categorical_to_numerical(X)

from sklearn.preprocessing import scale
X = scale(X)

regressor = LinearRegression()
regressor.fit(X, y)
