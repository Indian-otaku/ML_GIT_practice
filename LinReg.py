from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn.linear_model import LinearRegression

style.use('ggplot')

X = np.array([1,2,3,4,5,6,7,8,9])
y = np.array([3,4,4,7,8,9,7,8,9])

def best_fit_slope_and_intercept(X,y):
    m = ( ((mean(X)*mean(y)) - (mean(X*y)))/       # m = (X' * y' - (X * y)') / (X'^2 - (X^2)')
        ((mean(X)*mean(X)) - (mean(X*X))) )        # where ' represents taking mean

    b = mean(y) - (m * mean(X))  # From y = mx + b equation

    return m,b

def sum_of_squares(y_true,y_line):
    return sum((y_true-y_line)**2)

def coefficient_of_determination(y_true,y_line):
    y_mean_line = np.array([mean(y_true) for _ in y_true])
    plt.plot(X,y_mean_line)
    sum_of_squares_total = sum_of_squares(y_true,y_mean_line)
    sum_of_squares_error = sum_of_squares(y_true,y_line)

    return (1 - sum_of_squares_error / sum_of_squares_total)

m,b = best_fit_slope_and_intercept(X,y)
# print(m,b)
line = m*X + b
plt.scatter(X,y)
plt.plot(X,line,c='y')

# R2 score
print("R2 score: ",coefficient_of_determination(y,line))

# Lets predict the y values using some dummy X values 
X_pred = np.array([4.5,10,14])
y_pred = X_pred * m + b
print(y_pred)
plt.scatter(X_pred,y_pred,c='b')
