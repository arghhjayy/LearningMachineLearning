from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style

style.use('fivethirtyeight')

xs = np.array([1, 2, 3, 4, 5, 6], dtype=np.float64)
ys = np.array([5, 4, 6, 5, 6, 7], dtype=np.float64)

             # m #     #   b   #
def best_fit_slope_and_intercept(xs, ys):
    '''
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

m, b = best_fit_slope_and_intercept(xs, ys)
print(m, b)

regression_line = [(m*x) + b for x in xs]

#For x = 8:
predict_x = 8
#y would be m * x + b
predict_y = (m*predict_x) + b

plt.scatter(xs, ys)
plt.scatter(predict_x, predict_y, color='g')
plt.plot(xs, regression_line)
plt.show()