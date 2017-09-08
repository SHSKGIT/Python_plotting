#!/usr/bin/env python
# coding=utf-8


import numpy as np
import matplotlib
matplotlib.rc('xtick', labelsize=20)
matplotlib.rc('ytick', labelsize=20)
import matplotlib.pyplot as plt

import csv

httpMethod="GET"  # POST or GET
threadNumber=20
intervals=1000
sampleNumber=100


filename_1='/Users/JerryWang/Desktop/Experiment_%s_withDatabase_csv/%s_%dthreads_%dms_%dsamples_1.csv' % (httpMethod, httpMethod, threadNumber, intervals, sampleNumber)
filename_2='/Users/JerryWang/Desktop/Experiment_%s_withDatabase_csv/%s_%dthreads_%dms_%dsamples_2.csv' % (httpMethod, httpMethod, threadNumber, intervals, sampleNumber)
filename_3='/Users/JerryWang/Desktop/Experiment_%s_withDatabase_csv/%s_%dthreads_%dms_%dsamples_3.csv' % (httpMethod, httpMethod, threadNumber, intervals, sampleNumber)

'''
if threadNumber == 1:
    savePath='/Users/JerryWang/Desktop/Experiment_csv/%s_thread_%dms_intervals_%dsamples.png' % ('Single',intervals,sampleNumber)
else:
    savePath='/Users/JerryWang/Desktop/Experiment_csv/%d_threads_%dms_intervals_%dsamples.png' % (threadNumber,intervals,sampleNumber)
'''



with open(filename_1,'rb') as csvfile:
    reader = csv.DictReader(csvfile)
    column_threadGroup_1_raw = [row['threadName'] for row in reader]
#print column_threadGroup_1
#print type(column_threadGroup_1)
column_threadGroup_1=list(set(column_threadGroup_1_raw))
#print column_threadGroup_1
column_threadGroup_1.sort(key=lambda column_threadGroup_1_raw : int(column_threadGroup_1_raw.split('-')[-1]))
# get a sorted list ['Thread Group 1-1', 'Thread Group 1-2', 'Thread Group 1-3', 'Thread Group 1-4', 'Thread Group 1-5', 'Thread Group 1-6', 'Thread Group 1-7', 'Thread Group 1-8', 'Thread Group 1-9', 'Thread Group 1-10']
#print column_threadGroup_1


with open(filename_1,'rb') as csvfile:
    reader = csv.DictReader(csvfile)
    column_sampleNumber_1 = [int(row['sample#']) for row in reader]
#print type(column_sampleNumber[0])
#print column_sampleNumber[0]


with open(filename_1,'rb') as csvfile:
    reader = csv.DictReader(csvfile)
    column_each_1 = [int(row['elapsed']) for row in reader]
#print bool(len(column_each_1)==len(column_threadGroup_1_raw))

#dict_1={}
dict_thread_sorted_1={}
list_1=[]
for l in column_threadGroup_1:
    dict_thread_sorted_1[l]=[]   # create a list for each thread

#print dict_thread_sorted_1

for k in range(len(column_threadGroup_1_raw)):
    #key=k
    value=[column_threadGroup_1_raw[k], column_each_1[k]]
    #dict_1[key]=value

    for l in column_threadGroup_1:
        if value[0] == l:
            dict_thread_sorted_1[l].append(value[1])

#print dict_thread_sorted_1   # 不能对key排序，本来就是无序的
#  {'Thread Group 1-5': [239, 67, 81, 79, 75, 69, 37, 64, 49, 38, 45, 38, 42, 40, 39, 26, 41, 40, 29, 36, 59, 72, 91, 40, 41, 41, 25, 39, 39, 81, 29, 55, 64, 44, 25, 39, 49, 29, 26, 26, 25, 42, 55, 40, 25, 39, 42, 40, 39, 35, 41, 26, 42, 42, 43, 42, 28, 26, 43, 41, 26, 42, 54, 39, 36, 38, 39, 25, 40, 46, 43, 50, 47, 40, 41, 43, 39, 42, 40, 34, 26, 54, 42, 29, 40, 43, 39, 37, 40, 42, 47, 42, 50, 27, 46, 39, 41, 116, 48, 50], 'Thread Group 1-4': [216, 80, 99, 79, 69, 70, 41, 76, 50, 47, 55, 44, 43, 46, 61, 41, 44, 48, 29, 146, 67, 67, 42, 41, 40, 41, 29, 25, 39, 38, 41, 25, 30, 42, 25, 25, 57, 28, 27, 25, 26, 41, 54, 27, 27, 42, 34, 49, 43, 26, 25, 27, 39, 41, 40, 40, 26, 26, 39, 40, 25, 39, 37, 43, 30, 57, 41, 25, 29, 30, 39, 45, 52, 45, 57, 52, 30, 45, 45, 29, 36, 43, 42, 27, 38, 84, 54, 26, 38, 26, 31, 43, 38, 29, 31, 42, 29, 96, 55, 49], 'Thread Group 1-3': [201, 74, 88, 69, 58, 47, 27, 38, 39, 37, 41, 41, 39, 38, 40, 37, 40, 41, 27, 30, 73, 73, 72, 42, 40, 40, 55, 43, 42, 137, 26, 39, 40, 38, 25, 37, 41, 38, 26, 26, 28, 38, 39, 39, 25, 39, 39, 39, 38, 40, 37, 26, 40, 38, 39, 40, 35, 25, 39, 37, 25, 38, 39, 37, 38, 38, 38, 27, 60, 43, 40, 40, 38, 39, 40, 41, 38, 41, 39, 41, 28, 43, 38, 27, 41, 38, 42, 37, 38, 39, 39, 39, 27, 53, 43, 40, 53, 44, 50, 68], 'Thread Group 1-2': [185, 60, 42, 37, 40, 40, 56, 38, 37, 37, 41, 39, 38, 44, 41, 41, 39, 41, 30, 30, 67, 66, 69, 39, 41, 41, 40, 37, 37, 38, 38, 38, 38, 28, 37, 37, 37, 25, 60, 54, 37, 38, 38, 36, 26, 39, 38, 39, 26, 38, 38, 43, 40, 39, 38, 42, 38, 25, 38, 38, 26, 38, 36, 38, 39, 37, 34, 37, 38, 37, 39, 39, 26, 38, 27, 38, 41, 39, 30, 39, 41, 38, 28, 28, 41, 27, 26, 38, 30, 38, 26, 27, 57, 40, 38, 27, 31, 25, 26, 39], 'Thread Group 1-1': [41, 38, 42, 38, 27, 42, 28, 37, 25, 37, 40, 38, 41, 39, 41, 39, 39, 40, 57, 40, 100, 58, 69, 64, 44, 40, 40, 39, 25, 39, 38, 38, 26, 26, 37, 41, 25, 26, 26, 38, 37, 38, 46, 25, 26, 39, 26, 26, 26, 38, 27, 26, 26, 39, 40, 51, 38, 45, 43, 25, 46, 41, 34, 35, 46, 29, 39, 38, 32, 30, 25, 26, 26, 26, 26, 39, 41, 28, 27, 41, 27, 26, 26, 38, 41, 27, 25, 37, 44, 42, 25, 26, 40, 26, 27, 26, 26, 26, 26, 38]}
#print '\n\n'
list_thread_sorted_1=sorted(dict_thread_sorted_1.iteritems(), key=lambda k:int(k[0].split('-')[-1]))  # 排序后是一个list
#print list_thread_sorted_1

'''
dict_thread_sorted_1_keys=dict_thread_sorted_1.keys()  # ['Thread Group 1-5', 'Thread Group 1-4', 'Thread Group 1-3', 'Thread Group 1-2', 'Thread Group 1-1']
dict_thread_sorted_1_new=list(set(dict_thread_sorted_1_keys))
dict_thread_sorted_1_new.sort(key=lambda dict_thread_sorted_1_keys : int(dict_thread_sorted_1_keys.split('-')[-1]))
print dict_thread_sorted_1_new  # ['Thread Group 1-1', 'Thread Group 1-2', 'Thread Group 1-3', 'Thread Group 1-4', 'Thread Group 1-5']
'''
'''
dict_1={}
for d in range(len(list_thread_sorted_1)):
    dict_1[list_thread_sorted_1[d][0]]=list_thread_sorted_1[d][1]
print dict_1
'''

'''
y_1=[]
for y in range(len(list_thread_sorted_1)):
    y_1 += list_thread_sorted_1[y][1]

print '\n\n'
print y_1  # a list
'''

#dict_1=dict(map(lambda x,y : [x,y], key, value))


#column_average = [row2['average'] for row2 in reader]
'''
with open(filename_1,'rb') as csvfile:
    reader = csv.DictReader(csvfile)
    column_average_1 = [float(row['average']) for row in reader]
'''

#-------------------------------------------------

with open(filename_2,'rb') as csvfile:
    reader = csv.DictReader(csvfile)
    column_threadGroup_2_raw = [row['threadName'] for row in reader]
column_threadGroup_2=list(set(column_threadGroup_2_raw))
column_threadGroup_2.sort(key=lambda column_threadGroup_2_raw : int(column_threadGroup_2_raw.split('-')[-1]))
#print column_threadGroup_2

'''
with open(filename_2,'rb') as csvfile:
    reader = csv.DictReader(csvfile)
    column_sampleNumber_2 = [int(row['sample#']) for row in reader]
#print type(column_sampleNumber[0])
#print column_sampleNumber[0]
'''

with open(filename_2,'rb') as csvfile:
    reader = csv.DictReader(csvfile)
    column_each_2 = [int(row['elapsed']) for row in reader]
#column_average = [row2['average'] for row2 in reader]
'''
with open(filename_2,'rb') as csvfile:
    reader = csv.DictReader(csvfile)
    column_average_2 = [float(row['average']) for row in reader]
'''

dict_thread_sorted_2={}
list_2=[]
for l in column_threadGroup_2:
    dict_thread_sorted_2[l]=[]

for k in range(len(column_threadGroup_2_raw)):
    #key=k
    value=[column_threadGroup_2_raw[k], column_each_2[k]]
    #dict_1[key]=value
    
    for l in column_threadGroup_2:
        if value[0] == l:
            dict_thread_sorted_2[l].append(value[1])

#print dict_thread_sorted_2
#print '\n\n'
list_thread_sorted_2=sorted(dict_thread_sorted_2.iteritems(), key=lambda k:int(k[0].split('-')[-1]))
#print '\n\n'
#print list_thread_sorted_2
'''
y_2=[]
for y in range(len(list_thread_sorted_2)):
    y_2 += list_thread_sorted_2[y][1]
'''
#print y_2


#-------------------------------------------------

with open(filename_3,'rb') as csvfile:
    reader = csv.DictReader(csvfile)
    column_threadGroup_3_raw = [row['threadName'] for row in reader]
column_threadGroup_3=list(set(column_threadGroup_3_raw))
column_threadGroup_3.sort(key=lambda column_threadGroup_3_raw : int(column_threadGroup_3_raw.split('-')[-1]))
#print column_threadGroup_3


'''
with open(filename_3,'rb') as csvfile:
    reader = csv.DictReader(csvfile)
    column_sampleNumber_3 = [int(row['sample#']) for row in reader]
#print type(column_sampleNumber[0])
#print column_sampleNumber[0]
'''

with open(filename_3,'rb') as csvfile:
    reader = csv.DictReader(csvfile)
    column_each_3 = [int(row['elapsed']) for row in reader]
#column_average = [row2['average'] for row2 in reader]
'''
with open(filename_3,'rb') as csvfile:
    reader = csv.DictReader(csvfile)
    column_average_3 = [float(row['average']) for row in reader]
'''

dict_thread_sorted_3={}
list_3=[]
for l in column_threadGroup_3:
    dict_thread_sorted_3[l]=[]

for k in range(len(column_threadGroup_3_raw)):
    #key=k
    value=[column_threadGroup_3_raw[k], column_each_3[k]]
    #dict_1[key]=value
    
    for l in column_threadGroup_3:
        if value[0] == l:
            dict_thread_sorted_3[l].append(value[1])

#print dict_thread_sorted_3
#print '\n\n'
list_thread_sorted_3=sorted(dict_thread_sorted_3.iteritems(), key=lambda k:int(k[0].split('-')[-1]))
#print '\n\n'
#print list_thread_sorted_3
'''
y_3=[]
for y in range(len(list_thread_sorted_3)):
    y_3 += list_thread_sorted_3[y][1]
'''
#print y_3






x_step=10  # 不变
y_step=100
x_min=min(column_sampleNumber_1) # 不变
y_min=0 # 不变
x_max=max(column_sampleNumber_1)/threadNumber  # 不变
y_max=900
colorcode=['r', 'g', 'b', 'c', 'm', 'y', 'k', 'teal', 'pink', 'sienna', 'silver', 'violet', 'salmon', 'orange', 'ivory', 'tan', 'navy', 'orchid', 'olive', 'lime']
xy_label_size=30
subplot_title_size=30
xy_ticks_size=30


# 1st round
def f1(threadNum):  # 1st round中每一个thread
    #np.array=[]
    list_legend_threadNum=[]
    for t in range(threadNum):
        list_legend_threadNum.append('Thread %d' % (t+1))
        currentLabel='Thread %d' % threadNum
        #x=(1, len(column_sampleNumber_1)/threadNum, len(column_sampleNumber_1)/threadNum)
        #y=np.array(list_thread_sorted_1[t][1])
        ax1.plot(range(1,101), list_thread_sorted_1[t][1], color=colorcode[t], linestyle='solid', marker='', label=currentLabel)
    
    ax1.set_title('$1^{st} round$', fontsize=subplot_title_size)
    ax1.set_xticks(np.arange(x_min, x_max, x_step))
    #ax1.set_xticklabels(range(1,101), rotation = 45)
    ax1.set_yticks(np.arange(y_min,y_max,y_step))
    #ax1.set_yticklabels(range(20,900), rotation = 45)
    ax1.set_xlim((x_min,x_max))
    ax1.set_ylim((y_min, y_max))
    ax1.grid(True)
    ax1.tick_params(axis="both", which="major", labelsize=xy_ticks_size)
    
    
    '''
    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    '''
    
    '''
    legend_loc='upper right'
    ax1.legend(list_legend_threadNum, loc = legend_loc, fontsize = 20, shadow = True, bbox_to_anchor=(0.5, 0.5))
    '''

# 2nd round
def f2(threadNum):  # 1st round中每一个thread
    list_legend_threadNum=[]
    for t in range(threadNum):
        list_legend_threadNum.append('Thread %d' % (t+1))
        currentLabel='Thread %d' % threadNum
        #x=(1, len(column_sampleNumber_1)/threadNum, len(column_sampleNumber_1)/threadNum)
        #y=np.array(list_thread_sorted_1[t][1])
        ax2.plot(range(1,101), list_thread_sorted_2[t][1], color=colorcode[t], linestyle='solid', marker='', label=currentLabel)
    
    ax2.set_title('$2^{nd} round$', fontsize=subplot_title_size)
    ax2.set_xticks(np.arange(x_min, x_max, x_step))
    #ax2.set_xticklabels(range(1,101), rotation = 45)
    ax2.set_yticks(np.arange(y_min,y_max,y_step))
    #ax2.set_yticklabels(range(20,900), rotation = 45)
    ax2.set_xlim((x_min,x_max))
    ax2.set_ylim((y_min,y_max))
    ax2.grid(True)
    ax2.tick_params(axis="both", which="major", labelsize=xy_ticks_size)

    '''
    box = ax2.get_position()
    ax2.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    
    legend_loc='upper left'
    ax2.legend(list_legend_threadNum, loc = legend_loc, fontsize = 20, shadow = True, bbox_to_anchor=(1.02, 0.6))  # box_to_anchor被赋予的二元组中，第一个数值用于控制legend的左右移动，值越大越向右边移动，第二个数值用于控制legend的上下移动，值越大，越向上移动    (1.02, 0.8)
    '''

# 3rd round
def f3(threadNum):  # 1st round中每一个thread
    list_legend_threadNum=[]
    for t in range(threadNum):
        list_legend_threadNum.append('Thread %d' % (t+1))
        currentLabel='Thread %d' % threadNum
        #x=(1, len(column_sampleNumber_1)/threadNum, len(column_sampleNumber_1)/threadNum)
        #y=np.array(list_thread_sorted_1[t][1])
        ax3.plot(range(1,101), list_thread_sorted_3[t][1], color=colorcode[t], linestyle='solid', marker='', label=currentLabel)
    
    ax3.set_title('$3^{rd} round$', fontsize=subplot_title_size)
    ax3.set_xticks(np.arange(x_min, x_max, x_step))
    #ax3.set_xticklabels(range(1,101), rotation = 45)
    ax3.set_yticks(np.arange(y_min,y_max,y_step))
    #ax3.set_yticklabels(range(20,900), rotation = 45)
    ax3.set_xlim((x_min,x_max))
    ax3.set_ylim((y_min,y_max))
    ax3.grid(True)
    ax3.tick_params(axis="both", which="major", labelsize=xy_ticks_size)

    '''
    box = ax3.get_position()
    ax3.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    '''
    
    '''
    legend_loc='lower center'
    ax3.legend(list_legend_threadNum, loc = legend_loc, fontsize = 20, shadow = True, bbox_to_anchor=(0.5, 0.5))
    '''




fig = plt.figure()
ax1 = fig.add_subplot(311)
f1(threadNumber)
ax2 = fig.add_subplot(312, sharex=ax1, sharey=ax1)
f2(threadNumber)
ax3 = fig.add_subplot(313, sharex=ax1, sharey=ax1)
f3(threadNumber)
#fig, ax = plt.subplots(nrows=3, ncols=1)

'''
ax1=plt.subplot(311)
ax2=plt.subplot(312)
ax3=plt.subplot(313)
'''

'''
#ax=fig.add_subplot(111)
ax1=fig.add_subplot(311)
ax2=fig.add_subplot(312)
ax3=fig.add_subplot(313)
'''

chartTitle=''
if threadNumber == 1:
    if httpMethod == 'POST':
        chartTitle="%s thread %s to middleware at %dms intervals" % ('Single', httpMethod, intervals)
    else:
        chartTitle="%s thread %s from middleware at %dms intervals" % ('Single', httpMethod, intervals)
else:
    if httpMethod == 'POST':
        chartTitle="%d threads %s to middleware at %dms intervals" % (threadNumber, httpMethod, intervals)
    else:
        chartTitle="%d threads %s from middleware at %dms intervals" % (threadNumber, httpMethod, intervals)


fig.suptitle(chartTitle, fontsize=30)



ax3.set_xlabel("Index of samples (%d samples/thread)" % (sampleNumber), size=xy_label_size)
ax2.set_ylabel("Sample round trip time (ms)", size=xy_label_size)


'''
plt.xlabel("Index of samples (%d samples/thread)" % (sampleNumber), size=20)
plt.ylabel("Sample round trip time (ms)", size=20)
plt.grid(True)
'''
#ax1=plt.subplot(311)
#ax1=fig.add_subplot(311)
#ax1.plot([(1, 2), (3, 4)], [(4, 3), (2, 3)])
#f1(threadNumber)
#ax2=plt.subplot(312)
#ax2=fig.add_subplot(312)
#f2(threadNumber)
#ax3=plt.subplot(313)
#ax3=fig.add_subplot(313)
#f3(threadNumber)


'''
plt.plot(column_sampleNumber_1, y_1, 'r-')
#plt.plot(column_sampleNumber_1,column_average_1, 'r')

plt.plot(column_sampleNumber_1, y_2, 'g-')
#plt.plot(column_sampleNumber_1,column_average_2, 'g')

plt.plot(column_sampleNumber_1, y_3, 'b-')
#plt.plot(column_sampleNumber_1,column_average_3, 'b')
'''

'''
plt.xticks(np.arange(min(column_sampleNumber_1), max(column_sampleNumber_1)+1, 20), rotation = 45) #step默认就是1
#plt.yticks(np.arange(min(column_each), max(column_each)))
#plt.yticks(np.arange(min(column_each_1)-5, max(column_each_1)+5, 5))
plt.yticks(np.arange(20,900,20))

plt.xlim((1,len(column_sampleNumber_1))) #设置坐标轴范围
#plt.ylim((min(column_each_1)-5,max(column_each_1)+5))
plt.ylim((20,900))
'''


'''
tx=1
ty=-50
txt='Thread 1'
plt.text(tx, ty, txt, fontsize = 15, verticalalignment = "top", horizontalalignment = "left")
#plt.annotate('', xy=(tx,0), xytext=(tx,ty), arrowprops=dict(arrowstyle="->",connectionstyle="arc3"))
'''

'''
for t in range(len(column_threadGroup_1)):
    tx=(sampleNumber*(t))
    ty=-50
    txt='Thread %d' % int(column_threadGroup_1[t].split('-')[-1])
    plt.text(tx, ty, txt, fontsize = 15, verticalalignment = "top", horizontalalignment = "left", style='italic', bbox={'facecolor':'red', 'alpha':0.5, 'pad':10})
'''



#plt.legend(['each (1st)','average (1st) = '+str(column_average_1[0]),'each (2nd)','average (2nd) = '+str(column_average_2[0]),'each (3rd)','average (3rd) = '+str(column_average_3[0])], loc = 'upper right', fontsize = 'large', shadow = True)
#legend_loc='best'
'''
    right
    center left
    upper right
    lower right
    best
    center
    lower left
    center right
    upper left
    upper center
    lower center
'''
#plt.legend(['1st round', '2nd round', '3rd round'], loc = legend_loc, fontsize = 20, shadow = True)
#plt.savefig(savePath,dpi=1000)  #一定要在show()之前


plt.show()










