# vamshi143
In [1]:
import pandas as pd
import numpy as np
concrete_data = pd.read_csv('concrete_data.csv')
concrete_data.head()
Out[1]:
Cement	Blast Furnace Slag	Fly Ash	Water	Superplasticizer	Coarse Aggregate	Fine Aggregate	Age	Strength
0	540.0	0.0	0.0	162.0	2.5	1040.0	676.0	28	79.99
1	540.0	0.0	0.0	162.0	2.5	1055.0	676.0	28	61.89
2	332.5	142.5	0.0	228.0	0.0	932.0	594.0	270	40.27
3	332.5	142.5	0.0	228.0	0.0	932.0	594.0	365	41.05
4	198.6	132.4	0.0	192.0	0.0	978.4	825.5	360	44.30
In [2]:
#Let's check how many data points we have
concrete_data.shape
Out[2]:
(1030, 9)
In [3]:
concrete_data.describe()
Out[3]:
Cement	Blast Furnace Slag	Fly Ash	Water	Superplasticizer	Coarse Aggregate	Fine Aggregate	Age	Strength
count	1030.000000	1030.000000	1030.000000	1030.000000	1030.000000	1030.000000	1030.000000	1030.000000	1030.000000
mean	281.167864	73.895825	54.188350	181.567282	6.204660	972.918932	773.580485	45.662136	35.817961
std	104.506364	86.279342	63.997004	21.354219	5.973841	77.753954	80.175980	63.169912	16.705742
min	102.000000	0.000000	0.000000	121.800000	0.000000	801.000000	594.000000	1.000000	2.330000
25%	192.375000	0.000000	0.000000	164.900000	0.000000	932.000000	730.950000	7.000000	23.710000
50%	272.900000	22.000000	0.000000	185.000000	6.400000	968.000000	779.500000	28.000000	34.445000
75%	350.000000	142.950000	118.300000	192.000000	10.200000	1029.400000	824.000000	56.000000	46.135000
max	540.000000	359.400000	200.100000	247.000000	32.200000	1145.000000	992.600000	365.000000	82.600000
In [4]:
concrete_data.isnull().sum()
Out[4]:
Cement                0
Blast Furnace Slag    0
Fly Ash               0
Water                 0
Superplasticizer      0
Coarse Aggregate      0
Fine Aggregate        0
Age                   0
Strength              0
dtype: int64
In [5]:
#Split data into predictors and target
concrete_data_columns = concrete_data.columns
predictors = concrete_data[concrete_data_columns[concrete_data_columns != 'Strength']] # all columns except Strength
target = concrete_data['Strength'] # Strength column
predictors.head()
Out[5]:
Cement	Blast Furnace Slag	Fly Ash	Water	Superplasticizer	Coarse Aggregate	Fine Aggregate	Age
0	540.0	0.0	0.0	162.0	2.5	1040.0	676.0	28
1	540.0	0.0	0.0	162.0	2.5	1055.0	676.0	28
2	332.5	142.5	0.0	228.0	0.0	932.0	594.0	270
3	332.5	142.5	0.0	228.0	0.0	932.0	594.0	365
4	198.6	132.4	0.0	192.0	0.0	978.4	825.5	360
In [6]:
target.head()
Out[6]:
0    79.99
1    61.89
2    40.27
3    41.05
4    44.30
Name: Strength, dtype: float64
In [7]:
n_cols = predictors.shape[1] # number of predictors
n_cols
Out[7]:
8
In [8]:
#Import Keras
#Let's go ahead and import the Keras library
import keras
from keras.models import Sequential
from keras.layers import Dense
# define regression model
def regression_model():
    # create model
    model = Sequential()
    model.add(Dense(10, activation='relu', input_shape=(n_cols,)))
    model.add(Dense(1))
    # compile model
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model
Using TensorFlow backend.
In [9]:
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(predictors, target, test_size=0.3, random_state=42)
In [10]:
#Train and Test the Network
model = regression_model()
epochs = 50
model.fit(X_train, y_train, epochs=epochs, verbose=1)
Epoch 1/50
721/721 [==============================] - 1s 1ms/step - loss: 38389.9439
Epoch 2/50
721/721 [==============================] - 0s 135us/step - loss: 11232.6601
Epoch 3/50
721/721 [==============================] - 0s 116us/step - loss: 8290.0058
Epoch 4/50
721/721 [==============================] - 0s 122us/step - loss: 7146.0391
Epoch 5/50
721/721 [==============================] - 0s 111us/step - loss: 6214.1213
Epoch 6/50
721/721 [==============================] - 0s 169us/step - loss: 5464.9764
Epoch 7/50
721/721 [==============================] - 0s 58us/step - loss: 4836.3731
Epoch 8/50
721/721 [==============================] - 0s 50us/step - loss: 4279.7484
Epoch 9/50
721/721 [==============================] - 0s 44us/step - loss: 3805.8514
Epoch 10/50
721/721 [==============================] - 0s 51us/step - loss: 3419.8998
Epoch 11/50
721/721 [==============================] - 0s 48us/step - loss: 3097.3917
Epoch 12/50
721/721 [==============================] - 0s 51us/step - loss: 2840.7383
Epoch 13/50
721/721 [==============================] - 0s 56us/step - loss: 2601.1827
Epoch 14/50
721/721 [==============================] - 0s 55us/step - loss: 2396.0489
Epoch 15/50
721/721 [==============================] - 0s 55us/step - loss: 2216.3593
Epoch 16/50
721/721 [==============================] - 0s 53us/step - loss: 2051.1676
Epoch 17/50
721/721 [==============================] - 0s 50us/step - loss: 1902.6965
Epoch 18/50
721/721 [==============================] - 0s 53us/step - loss: 1761.7760
Epoch 19/50
721/721 [==============================] - 0s 52us/step - loss: 1629.8577
Epoch 20/50
721/721 [==============================] - 0s 91us/step - loss: 1509.2655
Epoch 21/50
721/721 [==============================] - 0s 86us/step - loss: 1398.5865
Epoch 22/50
721/721 [==============================] - 0s 86us/step - loss: 1300.5848
Epoch 23/50
721/721 [==============================] - 0s 111us/step - loss: 1208.2437
Epoch 24/50
721/721 [==============================] - 0s 108us/step - loss: 1128.5711
Epoch 25/50
721/721 [==============================] - 0s 100us/step - loss: 1058.7929
Epoch 26/50
721/721 [==============================] - 0s 119us/step - loss: 993.9317
Epoch 27/50
721/721 [==============================] - 0s 61us/step - loss: 937.1261
Epoch 28/50
721/721 [==============================] - 0s 58us/step - loss: 885.4081
Epoch 29/50
721/721 [==============================] - 0s 53us/step - loss: 837.6655
Epoch 30/50
721/721 [==============================] - 0s 50us/step - loss: 795.4289
Epoch 31/50
721/721 [==============================] - 0s 54us/step - loss: 758.7758
Epoch 32/50
721/721 [==============================] - 0s 53us/step - loss: 723.0673
Epoch 33/50
721/721 [==============================] - 0s 53us/step - loss: 691.2067
Epoch 34/50
721/721 [==============================] - 0s 53us/step - loss: 661.5906
Epoch 35/50
721/721 [==============================] - 0s 51us/step - loss: 634.7547
Epoch 36/50
721/721 [==============================] - 0s 53us/step - loss: 610.3779
Epoch 37/50
721/721 [==============================] - 0s 50us/step - loss: 586.5449
Epoch 38/50
721/721 [==============================] - 0s 50us/step - loss: 564.0321
Epoch 39/50
721/721 [==============================] - 0s 58us/step - loss: 544.4648
Epoch 40/50
721/721 [==============================] - 0s 64us/step - loss: 525.7497
Epoch 41/50
721/721 [==============================] - 0s 66us/step - loss: 508.8819
Epoch 42/50
721/721 [==============================] - 0s 66us/step - loss: 494.6443
Epoch 43/50
721/721 [==============================] - 0s 66us/step - loss: 479.2140
Epoch 44/50
721/721 [==============================] - 0s 66us/step - loss: 464.3036
Epoch 45/50
721/721 [==============================] - 0s 66us/step - loss: 449.5518
Epoch 46/50
721/721 [==============================] - 0s 69us/step - loss: 436.4871
Epoch 47/50
721/721 [==============================] - 0s 66us/step - loss: 423.9594
Epoch 48/50
721/721 [==============================] - 0s 64us/step - loss: 413.0417
Epoch 49/50
721/721 [==============================] - 0s 68us/step - loss: 401.7366
Epoch 50/50
721/721 [==============================] - 0s 64us/step - loss: 391.2751
Out[10]:
<keras.callbacks.callbacks.History at 0x22b0fcea5c8>
In [11]:
loss_val = model.evaluate(X_test, y_test)
y_pred = model.predict(X_test)
loss_val
309/309 [==============================] - 0s 604us/step
Out[11]:
428.9487768870727
In [12]:
from sklearn.metrics import mean_squared_error
mean_square_error = mean_squared_error(y_test, y_pred)
mean = np.mean(mean_square_error)
standard_deviation = np.std(mean_square_error)
print(mean, standard_deviation)
428.94877445915415 0.0
In [13]:
"""
Now we need to compute the mean squared error between the predicted concrete strength and the actual concrete strength.

Let's import the mean_squared_error function from Scikit-learn."""
total_mean_squared_errors = 50
epochs = 50
mean_squared_errors = []
for i in range(0, total_mean_squared_errors):
    X_train, X_test, y_train, y_test = train_test_split(predictors, target, test_size=0.3, random_state=i)
    model.fit(X_train, y_train, epochs=epochs, verbose=0)
    MSE = model.evaluate(X_test, y_test, verbose=0)
    print("MSE "+str(i+1)+": "+str(MSE))
    y_pred = model.predict(X_test)
    mean_square_error = mean_squared_error(y_test, y_pred)
    mean_squared_errors.append(mean_square_error)
"""
Create a list of 50 mean squared errors and report mean and the standard deviation of the mean squared errors."""
mean_squared_errors = np.array(mean_squared_errors)
mean = np.mean(mean_squared_errors)
standard_deviation = np.std(mean_squared_errors)

print('\n')
print("Below is the mean and standard deviation of " +str(total_mean_squared_errors) + " mean squared errors without normalized data. Total number of epochs for each training is: " +str(epochs) + "\n")
print("Mean: "+str(mean))
print("Standard Deviation: "+str(standard_deviation))
MSE 1: 135.31365376691602
MSE 2: 112.60569059501574
MSE 3: 87.3341740975488
MSE 4: 87.83425584811609
MSE 5: 81.34794004372408
MSE 6: 70.66509548051458
MSE 7: 75.60910234173525
MSE 8: 60.742101206362825
MSE 9: 64.17890001809327
MSE 10: 65.03571112179061
MSE 11: 60.79057114486941
MSE 12: 57.88739126014092
MSE 13: 68.92116849090675
MSE 14: 66.57615376444696
MSE 15: 61.64328557233594
MSE 16: 48.764096973011796
MSE 17: 54.18908358083188
MSE 18: 58.98746300360917
MSE 19: 48.334911976045774
MSE 20: 56.488057269247605
MSE 21: 49.131465825448146
MSE 22: 50.26347246139181
MSE 23: 44.006750829011494
MSE 24: 50.69993897471999
MSE 25: 46.97702820169887
MSE 26: 49.20486504514626
MSE 27: 48.2579986547575
MSE 28: 41.77312862680182
MSE 29: 56.59504161452013
MSE 30: 46.40752736804555
MSE 31: 46.622649776125414
MSE 32: 37.71383863282435
MSE 33: 43.2161382286294
MSE 34: 45.434861933143395
MSE 35: 48.00392210136339
MSE 36: 47.69304085395097
MSE 37: 45.75465921445186
MSE 38: 44.44095478243041
MSE 39: 42.70060902814649
MSE 40: 39.588684100549195
MSE 41: 45.58031794477049
MSE 42: 42.07940076315673
MSE 43: 42.51392466967931
MSE 44: 48.44099855885922
MSE 45: 44.915412162114116
MSE 46: 44.9094054089395
MSE 47: 44.53609964685533
MSE 48: 47.75831353394345
MSE 49: 45.658998322718354
MSE 50: 49.18928493265195


Below is the mean and standard deviation of 50 mean squared errors without normalized data. Total number of epochs for each training is: 50

Mean: 56.06635062330669
Standard Deviation: 18.304672093436402
