from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style

style.use('fivethirtyeight')

xs = np.array([1, 2, 3, 4, 5, 6], dtype=np.float64)
ys = np.array([5, 4, 6, 5, 6, 7], dtype=np.float64)

             # m #     #   b   #
def best_fit_slope_and_intercept(xs, ys):
    '''s_
    Calculate m and b
        __ __   _____
    m = xs*ys - xs*ys
        _____________
        __     _____
        xs^2 - xs**2
    '''
    m = (((mean(xs) * mean(ys)) - mean(xs*ys)) / ((mean(xs) ** 2) - mean(xs**2)))
    b = mean(ys) - m*mean(xs)
    return m, b

# Squared error (SE) = (point_at_line - original_point) ^ 2
def squared_error(ys_orig, ys_line):
    return sum((ys_line - ys_orig) ** 2)

# Coefficient of regression (r^2) is the value by which we determine
# how well our line fits on the graph. Higher the value, greater the
# line fits
# Formula:
# r ^ 2 = 1 - SquaredError(y_cap) / SquaredError(y_mean)
# Good r ^ 2 value : 0.8

def get_coefficient_of_determination(ys_orig, ys_line):
    y_mean_line = [mean(ys_orig) for y in ys_orig]
    squared_error_regr = squared_error(ys_orig, ys_line)
    squared_error_y_mean = squared_error(ys_orig, y_mean_line)
    return 1 - (squared_error_regr / squared_error_y_mean)

m, b = best_fit_slope_and_intercept(xs, ys)
print(m, b)

regression_line = [(m*x) + b for x in xs]

#For x = 8:
predict_x = 8
#y would be m * x + b
predict_y = (m*predict_x) + b

#Get the coefficient of determination
coefficient_of_determination = get_coefficient_of_determination(ys, regression_line)
print(coefficient_of_determination)

plt.scatter(xs, ys)
plt.scatter(predict_x, predict_y, color='g')
plt.plot(xs, regression_line)
plt.show()