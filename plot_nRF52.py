#!/usr/bin/env python
# coding=utf-8


import numpy as np
import matplotlib.pyplot as plt

import csv

#intervals=1000  # sampling every 1 second
sampleNumber=100
intervals=1000
travelType='roundtrip'  # 'WRITE', 'READ', 'roundtrip'
chipName='nRF52'

filename_1='/Users/JerryWang/Desktop/Experiment_BLE_csv/%s_%s_experiment_%dms_interval_1.csv' % (chipName, travelType, intervals)
filename_2='/Users/JerryWang/Desktop/Experiment_BLE_csv/%s_%s_experiment_%dms_interval_2.csv' % (chipName, travelType, intervals)
filename_3='/Users/JerryWang/Desktop/Experiment_BLE_csv/%s_%s_experiment_%dms_interval_3.csv' % (chipName, travelType, intervals)


# 1st round
with open(filename_1,'rb') as csvfile:
	reader = csv.DictReader(csvfile)
	column_sampleNumber_1 = [int(row['sample#']) for row in reader]

with open(filename_1,'rb') as csvfile:
    reader = csv.DictReader(csvfile)
    column_each_1 = [float(row['elapsed']) for row in reader]
'''
with open(filename_1,'rb') as csvfile:
    reader = csv.DictReader(csvfile)
    column_average_1 = [float(row['average']) for row in reader]
'''

#------------------------------------------------------------

# 2nd round
with open(filename_2,'rb') as csvfile:
	reader = csv.DictReader(csvfile)
	column_sampleNumber_2 = [int(row['sample#']) for row in reader]

with open(filename_2,'rb') as csvfile:
    reader = csv.DictReader(csvfile)
    column_each_2 = [float(row['elapsed']) for row in reader]
'''
with open(filename_2,'rb') as csvfile:
    reader = csv.DictReader(csvfile)
    column_average_2 = [float(row['average']) for row in reader]
'''

#------------------------------------------------------------

# 3rd round
with open(filename_3,'rb') as csvfile:
	reader = csv.DictReader(csvfile)
	column_sampleNumber_3 = [int(row['sample#']) for row in reader]

with open(filename_3,'rb') as csvfile:
    reader = csv.DictReader(csvfile)
    column_each_3 = [float(row['elapsed']) for row in reader]
'''
with open(filename_3,'rb') as csvfile:
    reader = csv.DictReader(csvfile)
    column_average_3 = [float(row['average']) for row in reader]
'''








x_step=10
y_step=20
x_min=min(column_sampleNumber_1)
y_min=120
x_max=max(column_sampleNumber_1)
y_max=460
xy_ticks_size=30







fig=plt.figure()

if travelType == 'WRITE':
    chartTitle='Central %s to peripheral at 1000ms intervals' % travelType
elif travelType == 'READ':
    chartTitle='Central %s from peripheral at 1000ms intervals' % travelType
else:
    chartTitle='Central %s to peripheral at 1000ms intervals' % travelType

plt.title(chartTitle, fontsize=xy_ticks_size)
plt.xlabel("Index of samples (%d samples/round)" % sampleNumber, fontsize=xy_ticks_size)
plt.ylabel("Sample %s time (ms)" % travelType, fontsize=xy_ticks_size)
plt.grid(True)

plt.plot(column_sampleNumber_1, column_each_1, 'r-')
#plt.plot(column_sampleNumber_1, column_average_1, 'r-')
plt.plot(column_sampleNumber_2, column_each_2, 'g-')
#plt.plot(column_sampleNumber_2, column_average_2, 'g-')
plt.plot(column_sampleNumber_3, column_each_3, 'b-')
#plt.plot(column_sampleNumber_3, column_average_3, 'b-')

plt.xticks(np.arange(x_min, x_max, x_step), rotation = 45, fontsize=xy_ticks_size) #step is 1 by default
plt.yticks(np.arange(y_min, y_max, y_step), fontsize=xy_ticks_size)

#plot = fig.add_subplot(111)
#plot.tick_params(axis="both", which="major", labelsize=xy_ticks_size)

plt.xlim((x_min,x_max))
plt.ylim((y_min,y_max))

plt.legend(['$1^{st} round$','$2^{nd} round$','$3^{rd} round$'], loc = 'upper right', fontsize = 30, shadow = True)

plt.show()





















