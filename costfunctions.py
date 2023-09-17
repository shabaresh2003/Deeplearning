import numpy as np
y_predicted = np.array([1,1,0,0,1])
y_actual = np.array([0.3,0.7,1,0,1])
def mean_abs_error(y_actual ,y_predicted):
    total_error = 0
    for yt , yp in zip(y_actual,y_predicted):
        total_error += abs(yt-yp)
    mae = total_error/len(y_actual)
    print("mae",mae)
    return  mae;

print("mean_abs_error",mean_abs_error(y_actual,y_predicted))
# Mean absolute error is the total error of the y_predicted and y_actual.





#Log losss function or Binary crossentropy
def log_loss(y_actual,y_predicted):
    epsilon = 1e-15
    y_predicted_new = [max(epsilon, i) for i in y_predicted]
    # log(0) is undefined so we convert  0 to near 0 value to calculate the binary cross entropy.
    y_predicted_new = [min(i, 1 - epsilon) for i in y_predicted_new]
    y_predicted_new = np.array(y_predicted_new)
    return -np.mean(y_actual*(np.log(y_predicted_new) + ((1 - y_actual) * (np.log(1 - y_predicted_new)))))
print("log_loss or binary cross entropy",log_loss(y_actual,y_predicted))


def mean_square_error(y_actual,y_predicted):
    total_error = 0
    for i in range(len(y_actual)):
        total_error += (y_actual[i] - y_predicted[i])**2
    return total_error/len(y_actual)


print("mean_square_error",mean_square_error(y_actual,y_predicted))