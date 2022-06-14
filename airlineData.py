import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
col = ['FL_DATE','OP_CARRIER','ORIGIN','DEST','ARR_DELAY','DEP_DELAY']
# reading data using 2014 dataset
data2014 = pd.read_csv("/home/knez/Desktop/vsc_code/Airline data/2014.csv",header=0,encoding='utf-8',usecols=col)


# select arrival delay and departure delay columns
arr = data2014['ARR_DELAY'].values
dep = data2014['DEP_DELAY'].values

# conversion from 1 dimensions to 2
arr_del = arr.reshape(-1,1)
dep_del = dep.reshape(-1,1)


# using scipy SimpleImputer to change NaN values to mean value
imp = SimpleImputer(missing_values=np.NaN,strategy='mean')
arr_del = imp.fit_transform(arr_del)
dep_del = imp.fit_transform(dep_del)

# calculate and print mean and median value for arrival delay data
timeDel_mean = np.mean(arr_del)
timeDel_median = np.median(arr_del)

print("-"*15,"THIS IS FOR ARRIVAL (2014)","-"*15)
print("The mean value of delay time in minutes is: %.3f" %timeDel_mean)
print("The median value of delay time in minutes is: %.3f" %(timeDel_median))

# import scipy tools for skewness and kurtosis
from scipy.stats import skew
from scipy.stats import kurtosis

# calculate and print value of skewness and kurtosis for arrival delay data
print("The skewness of data is:%.3f" %skew(arr_del))
print("The kurtosis of data is:%.3f" %kurtosis(arr_del))

print("-"*15,"THIS IS FOR DEPARTURE (2014)","-"*15)

# calculate and print mean and median value for departure delay data
dep_del_mean = np.mean(dep_del)
dep_del_median = np.median(dep_del)

print("The mean value of delay time in minutes is: %.3f" %dep_del_mean)
print("The median value of delay time in minutes is: %.3f" %dep_del_median)

# calculate and print skewness and kurtosis for departure delay time
print("The skewness of data is:%.3f" %skew(dep_del))
print("The kurtosis of data is:%.3f" %kurtosis(dep_del))




###########################  THIS IS FOR 2015 YEAR AND WE DO THE SAME THING WE USED FOR 2014 ###################################

data2015 = pd.read_csv("/home/knez/Desktop/vsc_code/Airline data/2015.csv",header=0,encoding='utf-8',usecols=col)
arr15 = data2015['ARR_DELAY'].values
dep15 = data2015['DEP_DELAY'].values

arr_del15 = arr15.reshape(-1,1)
dep_del15 = dep15.reshape(-1,1)

arr_del15 = imp.fit_transform(arr_del15)
dep_del15 = imp.fit_transform(dep_del15)

a_del15 = np.mean(arr_del15)
a_del15_med = np.median(arr_del15)


d_del15 = np.mean(dep_del15)
d_del15_med = np.median(dep_del15)


print("-"*15,"THIS IS FOR ARRIVAL (2015)","-"*15)
print("The mean value of delay time in minutes is:%.3f" %(a_del15))
print("The median value of delay time in minutes is:%.3f" %(a_del15_med))
print("The skewness of arrival data is: %.3f" %skew(arr_del15))
print("The kurtosis of arrival data is: %.3f" %kurtosis(arr_del15))



print("-"*15,"THIS IS FOR DEPARTURE (2015)","-"*15)
print("The mean value of delay time in minutes is:%.3f" %(d_del15))
print("The median value of delay time in minutes is:%.3f" %(d_del15_med))
print("The skewness of departure data is: %.3f" %skew(dep_del15))
print("The kurtosis of departure data is: %.3f" %kurtosis(dep_del15))





################################### THIS IS FOR 2016 DATA ##########################################################
################################ WE DO THE SAME THING AS FOR 2014 #################################################

data2016 = pd.read_csv("/home/knez/Desktop/vsc_code/Airline data/2016.csv",header=0,encoding='utf-8',usecols=col)

arr16 = data2016['ARR_DELAY'].values
dep16 = data2016['DEP_DELAY'].values

arr16 = arr16.reshape(-1,1)
dep16 = dep16.reshape(-1,1)

arr16 = imp.fit_transform(arr16)
dep16 = imp.fit_transform(dep16)

arr_del16 = np.mean(arr16)
arr_del16_med = np.median(arr16)

print("-"*15,"THIS IS FOR ARRIVAL 2016","-"*15)
print("The mean value of delay time in minutes is: %.3f" %(arr_del16))
print("The median value of delay time in minutes is: %.3f" %(arr_del16_med))
print("The skewness value of delay time is: %.3f" %skew(arr16))
print("The kurtosis value of delay time is: %.3f" %kurtosis(arr16))

dep16_del = np.mean(dep16)
dep16_del_med = np.median(dep16)

print("-"*15,"THIS IS FOR DEPARTURE 2016","-"*15)
print("The mean value of delay time in minutes is: %.3f" %dep16_del)
print("The median value of delay time in minutes is: %.3f" %dep16_del_med)
print("The skewness value of delay time is: %.3f" %skew(dep16))
print("The kurtosis value of delay time is: %.3f" %kurtosis(dep16))













# making histograms for all years



import matplotlib.pyplot as plt
fig,ax= plt.subplots(nrows=3,ncols=2,figsize=(10,10))


# 2014 data 

ax[0,0].hist(arr_del,range=(-100,100))
ax[0,0].set_ylabel("Number of delayed flights")
ax[0,0].set_xlabel("Delay in minutes")
ax[0,0].set_title("Delays in arrival 2014")

ax[0,1].hist(dep_del,range=(-100,100),color='red')
ax[0,1].set_ylabel("Number of delayed flights")
ax[0,1].set_xlabel("Delay in minutes")
ax[0,1].set_title("Delays in departure 2014")


# 2015 data
ax[1,0].hist(arr_del15,range=(-100,100))
ax[1,0].set_ylabel("Number of delayed flights")
ax[1,0].set_xlabel("Delay in minutes")
ax[1,0].set_title("Delays in arrival 2015")

ax[1,1].hist(dep_del15,range=(-100,100),color='red')
ax[1,1].set_ylabel("Number of delayed flights")
ax[1,1].set_xlabel("Delay in minutes")
ax[1,1].set_title("Delays in departure 2015")



#fig,ax= plt.subplots(nrows=2,ncols=2,figsize=(10,10))
# 2016 data

ax[2,0].hist(arr16,range=(-100,100))
ax[2,0].set_ylabel("Number of delayed flights")
ax[2,0].set_xlabel("Delay in minutes")
ax[2,0].set_title("Delays in arrival 2016")


ax[2,1].hist(dep16,range=(-100,100),color='red')
ax[2,1].set_ylabel("Number of delayed flights")
ax[2,1].set_xlabel("Delay in minutes")
ax[2,1].set_title("Delays in departure 2016")


plt.tight_layout()
plt.show()


y2014 = np.mean(arr_del+dep_del)
y2015 = np.mean(arr_del15+dep_del15)
y2016 = np.mean(arr16+dep16)

y2014_med = np.median(arr_del+dep_del)
y2015_med = np.median(arr_del15+dep_del15)
y2016_med = np.median(arr16+dep16)



print("-"*15," MEAN VALUE BY YEAR","-"*15)
print("The mean delay time in minutes (2014): %.3f" %(y2014))
print("The mean delay time in minutes (2015): %.3f" %(y2015))
print("The mean delay time in minutes (2016): %.3f" %(y2016))


print("-"*15," MEDIAN VALUE BY YEAR","-"*15)
print("The median delay time in minutes (2014): %.3f" %(y2014_med))
print("The median delay time in minutes (2015): %.3f" %(y2015_med))
print("The median delay time in minutes (2016): %.3f" %(y2016_med))




fl_date = data2014['FL_DATE']
fl_date15 = data2015['FL_DATE']
fl_date16 = data2016['FL_DATE']

for i in range(len(fl_date)):
    tmp = fl_date[i]
    if tmp=='2014-04-01':
        q1 = i
    elif tmp=='2014-07-01':
        q2 = i
    elif tmp=='2014-10-01':
        q3= i

for i in range(len(fl_date15)):
    tmp = fl_date15[i]
    if tmp=='2015-04-01':
        q1_15 = i
    elif tmp=='2015-07-01':
        q2_15 = i
    elif tmp=='2015-10-01':
        q3_15 = i

for i in range(len(fl_date16)):
    tmp = fl_date16[i]
    if tmp=='2016-04-01':
        q1_16 = i
    elif tmp=='2016-07-01':
        q2_16 = i
    elif tmp=='2016-10-01':
        q3_16 = i


# calculate quartals for arrivals and departures in 2014


# selecting dates
quar1_a,quar1_d = arr_del[:q1],dep_del[:q1]
quar2_a,quar2_d = arr_del[q1:q2],dep_del[q1:q2]
quar3_a,quar3_d = arr_del[q2:q3],dep_del[q2:q3]
quar4_a,quar4_d = arr_del[q3],dep_del[q3]

################ THIS IS FOR ARRIVAL 2014 ##################

# calculate mean and median for quartal 1
qa1_mean = np.mean(quar1_a)
qa1_med = np.median(quar1_a)

# calculate mean and median for quartal 2
qa2_mean = np.mean(quar2_a)
qa2_med = np.median(quar2_a)

# calculate mean and median for quartal 3
qa3_mean = np.mean(quar3_a)
qa3_med = np.median(quar3_a)

# calculate mean and median for quartal 4
qa4_mean = np.mean(quar4_a)
qa4_med = np.median(quar4_a)

################ THIS IS FOR DEPARTURE 2014 #########################

# quartal 1 deparutre
qd1_mean = np.mean(quar1_d)
qd1_med = np.median(quar1_d)

#quartal 2 departure
qd2_mean = np.mean(quar2_d)
qd2_med= np.median(quar2_d)

#quartal 3 departure
qd3_mean = np.mean(quar3_d)
qd3_med = np.median(quar3_d)

# quartal 4 departure

qd4_mean = np.mean(quar4_d)
qd4_med = np.median(quar4_d)

quar2014a_mean = [qa1_mean,qa2_mean,qa3_mean,qa4_mean]
quar2014d_mean = [qd1_mean,qd2_mean,qd3_mean,qd4_mean]

quar2014a_med =[qa1_med,qa2_med,qa3_med,qa4_med]
quar2014d_med = [qd1_med,qd2_med,qd3_med,qd4_med]



################ THIS IS FOR ARRIVAL 2015 ########################

quar1_a_15,quar1_d_15 = arr_del15[:q1_15],dep_del15[:q1_15]
quar2_a_15,quar2_d_15=arr_del15[q1_15:q2_15],dep_del15[q1_15:q2_15]
quar3_a_15,quar3_d_15 = arr_del15[q2_15:q3_15],dep_del15[q2_15:q3_15]
quar4_a_15,quar4_d_15 = arr_del15[q3_15:],dep_del15[q3_15:]

q1a15_mean,q1a15_med = np.mean(quar1_a_15),np.median(quar1_a_15)
q1d15_mean,q1d15_med = np.mean(quar1_d_15),np.median(quar1_d_15)
q2a15_mean,q2a15_med = np.mean(quar2_a_15),np.median(quar2_a_15)
q2d15_mean,q2d15_med = np.mean(quar3_d_15),np.median(quar3_d_15)
q3a15_mean,q3a15_med = np.mean(quar3_a_15),np.median(quar3_a_15)
q3d15_mean,q3d15_med = np.mean(quar3_d_15),np.median(quar3_d_15)
q4a15_mean,q4a15_med = np.mean(quar4_a_15),np.median(quar4_a_15)
q4d15_mean,q4d15_med = np.mean(quar4_d_15),np.median(quar4_d_15)


quar2015a_mean = [q1a15_mean,q2a15_mean,q3a15_mean,q4a15_mean]
quar2015d_mean = [q1d15_mean,q2d15_mean,q3d15_mean,q4d15_mean]

quar2015a_med =[q1a15_med,q2a15_med,q3a15_med,q4a15_med]
quar2015d_med = [q1d15_med,q2d15_med,q3d15_med,q4d15_med]


################################# THIS IS FOR 2016 DATA ########################################


quar1_a_16,quar1_d_16 = arr16[:q1_16],dep16[:q1_16]
quar2_a_16,quar2_d_16=arr16[q1_16:q2_16],dep16[q1_16:q2_16]
quar3_a_16,quar3_d_16 = arr16[q2_16:q3_16],dep16[q2_16:q3_16]
quar4_a_16,quar4_d_16 = arr16[q3_16:],dep16[q3_16:]


q1a16_mean,q1a16_med = np.mean(quar1_a_16),np.median(quar1_a_16)
q1d16_mean,q1d16_med = np.mean(quar1_d_16),np.median(quar1_d_16)
q2a16_mean,q2a16_med = np.mean(quar2_a_16),np.median(quar2_a_16)
q2d16_mean,q2d16_med = np.mean(quar3_d_16),np.median(quar3_d_16)
q3a16_mean,q3a16_med = np.mean(quar3_a_16),np.median(quar3_a_16)
q3d16_mean,q3d16_med = np.mean(quar3_d_16),np.median(quar3_d_16)
q4a16_mean,q4a16_med = np.mean(quar4_a_16),np.median(quar4_a_16)
q4d16_mean,q4d16_med = np.mean(quar4_d_16),np.median(quar4_d_16)

quar2016a_mean = [q1a16_mean,q2a16_mean,q3a16_mean,q4a16_mean]
quar2016d_mean = [q1d16_mean,q2d16_mean,q3d16_mean,q4d16_mean]

quar2016a_med =[q1a16_med,q2a16_med,q3a16_med,q4a16_med]
quar2016d_med = [q1d16_med,q2d16_med,q3d16_med,q4d16_med]

x_axis = [1,2,3,4]
plot1 = plt.figure(1)
plt.plot(x_axis,quar2014a_mean,marker='o',color='blue',label='Mean arrival(2014)')
plt.plot(x_axis,quar2014a_med,marker='x',linestyle='--',color='blue',label='Median arrival(2014)')
plt.plot(x_axis,quar2015a_mean,marker='o',color='purple',label='Mean arrival(2015)')
plt.plot(x_axis,quar2015a_med,marker='x',linestyle='--',color='purple',label='Median arrival(2015)')
plt.plot(x_axis,quar2016a_mean,marker='o',color='gray',label='Mean arrival(2016)')
plt.plot(x_axis,quar2016a_med,marker='x',linestyle='--',color='gray',label='Median arrival(2016)')
plt.title("Mean and median in quartals (arrivals 2014,2015,2016)")
plt.xticks(x_axis)
plt.legend(loc="best")
plt.grid(axis='x',linewidth=2)
plt.xlabel("Quartals")
plt.ylabel("Values in minutes")

plot2 = plt.figure(2)
plt.plot(x_axis,quar2014d_mean,marker='o',color='red',label='Mean departure(2014)')
plt.plot(x_axis,quar2014d_med,marker='x',linestyle='--',color='red',label='Median departure(2014)')
plt.plot(x_axis,quar2015d_mean,marker='o',color='green',label='Mean departure(2015)')
plt.plot(x_axis,quar2015d_med,marker='x',linestyle='--',color='green',label='Median departure(2015)')
plt.plot(x_axis,quar2016d_mean,marker='o',color='black',label='Mean departure(2016)')
plt.plot(x_axis,quar2016d_med,marker='x',linestyle='--',color='black',label='Median departure(2016)')
plt.xticks(x_axis)
plt.legend(loc="best")
plt.title("Mean and median in quartals(departure 2014,2015,2016)")
plt.grid(axis='x',linewidth=2)
plt.xlabel("Quartals")
plt.ylabel("Values in minutes")
plt.tight_layout()
plt.show()

# SELECT INDEXES FOR VALUES SO WE CAN SEPARATE DATA BY MONTH
ind1,ind2,ind3,ind4,ind5,ind6,ind7,ind8,ind9,ind10,ind11,ind12 = [],[],[],[],[],[],[],[],[],[],[],[]


for i in range(len(fl_date)):
    tmp = fl_date[i]
    if tmp[5:7]=='01':
        ind1.append(i)
    elif tmp[5:7]=='02':
        ind2.append(i)
    elif tmp[5:7]=='03':
        ind3.append(i)
    elif tmp[5:7]=='04':
        ind4.append(i)
    elif tmp[5:7]=='05':
        ind5.append(i)
    elif tmp[5:7]=='06':
        ind6.append(i)
    elif tmp[5:7]=='07':
        ind7.append(i)
    elif tmp[5:7]=='08':
        ind8.append(i)
    elif tmp[5:7]=='09':
        ind9.append(i)
    elif tmp[5:7]=='10':
        ind10.append(i)
    elif tmp[5:7]=='11':
        ind11.append(i)
    elif tmp[5:7]=='12':
        ind12.append(i)


########  THIS IS FOR 2014 DATA IN MONTHS  ##################

m1_a,m1_d = arr_del[ind1],dep_del[ind1]
m2_a,m2_d = arr_del[ind2],dep_del[ind2]
m3_a,m3_d = arr_del[ind3],dep_del[ind3]
m4_a,m4_d = arr_del[ind4],dep_del[ind4]
m5_a,m5_d = arr_del[ind5],dep_del[ind5]
m6_a,m6_d = arr_del[ind6],dep_del[ind6]
m7_a,m7_d = arr_del[ind7],dep_del[ind7]
m8_a,m8_d = arr_del[ind8],dep_del[ind8]
m9_a,m9_d = arr_del[ind9],dep_del[ind9]
m10_a,m10_d = arr_del[ind10],dep_del[ind10]
m11_a,m11_d = arr_del[ind11],dep_del[ind11]
m12_a,m12_d = arr_del[ind12],dep_del[ind12]

# CALCULATING MEAN AND MEDIAN IN 2014
m1a_mean = np.mean(m1_a)
m1a_med = np.median(m1_a)
m1d_mean = np.mean(m1_d)
m1d_med = np.median(m1_d)

m2a_mean = np.mean(m2_a)
m2a_med = np.median(m2_a)
m2d_mean = np.mean(m2_d)
m2d_med = np.median(m2_d)

m3a_mean = np.mean(m3_a)
m3a_med = np.median(m3_a)
m3d_mean = np.mean(m3_d)
m3d_med = np.median(m3_d)

m4a_mean = np.mean(m4_a)
m4a_med = np.median(m4_a)
m4d_mean = np.mean(m4_d)
m4d_med = np.median(m4_d)

m5a_mean = np.mean(m5_a)
m5a_med = np.median(m5_a)
m5d_mean = np.mean(m5_d)
m5d_med = np.median(m5_d)

m6a_mean = np.mean(m6_a)
m6a_med = np.median(m6_a)
m6d_mean = np.mean(m6_d)
m6d_med = np.median(m6_d)

m7a_mean = np.mean(m7_a)
m7a_med = np.median(m7_a)
m7d_mean = np.mean(m7_d)
m7d_med = np.median(m7_d)

m8a_mean = np.mean(m8_a)
m8a_med = np.median(m8_a)
m8d_mean = np.mean(m8_d)
m8d_med = np.median(m8_d)

m9a_mean = np.mean(m9_a)
m9a_med = np.median(m9_a)
m9d_mean = np.mean(m9_d)
m9d_med = np.median(m9_d)

m10a_mean = np.mean(m10_a)
m10a_med = np.median(m10_a)
m10d_mean = np.mean(m10_d)
m10d_med = np.median(m10_d)

m11a_mean = np.mean(m11_a)
m11a_med = np.median(m11_a)
m11d_mean = np.mean(m11_d)
m11d_med = np.median(m11_d)

m12a_mean = np.mean(m12_a)
m12a_med = np.median(m12_a)
m12d_mean = np.mean(m12_d)
m12d_med = np.median(m12_d)

# STORE OUR RESULTS IN LISTS
month_mean_a = [m1a_mean,m2a_mean,m3a_mean,m4a_mean,m5a_mean,m6a_mean,m7a_mean,m8a_mean,m9a_mean,m10a_mean,m11a_mean,m12a_mean]
month_mean_d= [m1d_mean,m2d_mean,m3d_mean,m4d_mean,m5d_mean,m6d_mean,m7d_mean,m8d_mean,m9d_mean,m10d_mean,m11d_mean,m12d_mean]

month_med_a = [m1a_med,m2a_med,m3a_med,m4a_med,m5a_med,m6a_med,m7a_med,m8a_med,m9a_med,m10a_med,m11a_med,m12a_med]
month_med_d = [m1d_med,m2d_med,m3d_med,m4d_med,m5d_med,m6d_med,m7d_med,m8d_med,m9d_med,m10d_med,m11d_med,m12d_med]

# PLOT VALUES 
plt.plot(range(1,len(month_mean_a)+1),month_mean_a,color='blue',marker='o',label='Mean value by months (arrival)')
plt.plot(range(1,len(month_mean_d)+1),month_mean_d,color='red',marker='s',label='Mean value by months (departure)')
plt.plot(range(1,len(month_med_a)+1),month_med_a,color='blue',linestyle='--',marker='o',label='Median value by months (arrival)')
plt.plot(range(1,len(month_med_d)+1),month_med_d,color='red',linestyle='--',marker='s',label='Median value by months (departure)')
plt.xticks([1,2,3,4,5,6,7,8,9,10,11,12])
plt.grid()
plt.xlabel("Months")
plt.ylabel("Values in minutes")
plt.legend(loc='best')
plt.title("Mean and median value of delays in months (2014)")
plt.tight_layout()
plt.show()

del fl_date,fl_date15,fl_date16
############## CALCULATE MEAN AND MEDIAN FOR THE WEEK OF YEAR (2014) ##############################
    
from datetime import datetime
import datetime

dates = data2014['FL_DATE'].values
dates15 = data2015['FL_DATE'].values
dates16 = data2016['FL_DATE'].values


weeks,weeks15,weeks16=[],[],[]
days_num,days_num15,days_num16=[],[],[]

# GENERATING WEEK AND DAY NUMS FROM 2014

for i in range(len(dates)):
    tmp = dates[i]
    year =int(tmp[:4])
    mont = int(tmp[5:7])
    day = int(tmp[8:])
    tmp = datetime.date(year,mont,day)
    week_num = tmp.isocalendar()[1]
    day_num = tmp.isocalendar()[2]
    weeks.append(week_num)
    days_num.append(day_num)


for i in range(len(dates15)):
    tmp = dates15[i]
    year =int(tmp[:4])
    mont = int(tmp[5:7])
    day = int(tmp[8:])
    tmp = datetime.date(year,mont,day)
    week_num = tmp.isocalendar()[1]
    day_num = tmp.isocalendar()[2]
    weeks15.append(week_num)
    days_num15.append(day_num)

for i in range(len(dates16)):
    tmp = dates16[i]
    year =int(tmp[:4])
    mont = int(tmp[5:7])
    day = int(tmp[8:])
    tmp = datetime.date(year,mont,day)
    week_num = tmp.isocalendar()[1]
    day_num = tmp.isocalendar()[2]
    weeks16.append(week_num)
    days_num16.append(day_num)

# MAKING LIST FOR RESULTS MEAN AND MEDIAN
week_mean_arr = []
week_mean_dep = []

week_med_arr = []
week_med_dep = []

week_mean_arr15 = []
week_mean_dep15 = []

week_med_arr15 = []
week_med_dep15 = []

week_mean_arr16 = []
week_mean_dep16 = []

week_med_arr16 = []
week_med_dep16 = []

# WE FIND INDEXES WHERE WE HAVE CHANGES IN LIST WEEKS (1 ---> 2 etc) WE TAKE INDEXES FROM WEEK 2 BECAUSE FIRST WEEK IN JANUARY STARTS AT MONDAY 6.
ind = np.where(np.roll(weeks,1)!=weeks)[0]
ind15 = np.where(np.roll(weeks15,1)!=weeks15)[0]
ind16 = np.where(np.roll(weeks16,1)!=weeks16)[0]



# SELECTING DATA USING ind AND CALCULATE MEAN AND MEDIAN
for i in range(len(ind)-1):
    ind1=ind[i]
    ind2 = ind[i+1]
    tmp1 = arr_del[ind1:ind2]
    tmp2 = dep_del[ind1:ind2]

    week_mean_arr.append(round(np.mean(tmp1),2))
    week_mean_dep.append(round(np.mean(tmp2),2))

    week_med_arr.append(round(np.median(tmp1),2))
    week_med_dep.append(round(np.median(tmp2),2))


for i in range(len(ind15)-1):
    ind1=ind15[i]
    ind2 = ind15[i+1]
    tmp1 = arr_del15[ind1:ind2]
    tmp2 = dep_del15[ind1:ind2]

    week_mean_arr15.append(round(np.mean(tmp1),2))
    week_mean_dep15.append(round(np.mean(tmp2),2))

    week_med_arr15.append(round(np.median(tmp1),2))
    week_med_dep15.append(round(np.median(tmp2),2))

for i in range(len(ind16)-1):
    ind1=ind16[i]
    ind2 = ind16[i+1]
    tmp1 = arr16[ind1:ind2]
    tmp2 = dep16[ind1:ind2]

    week_mean_arr16.append(round(np.mean(tmp1),2))
    week_mean_dep16.append(round(np.mean(tmp2),2))

    week_med_arr16.append(round(np.median(tmp1),2))
    week_med_dep16.append(round(np.median(tmp2),2))

# SELECTING DAYS AND CALCULATING MEAN AND MEDIAN
mon,tue,wed,thu,fri,sat,sun = [],[],[],[],[],[],[]
mon_d,tue_d,wed_d,thu_d,fri_d,sat_d,sun_d = [],[],[],[],[],[],[]

mon15,tue15,wed15,thu15,fri15,sat15,sun15 = [],[],[],[],[],[],[]
mon_d15,tue_d15,wed_d15,thu_d15,fri_d15,sat_d15,sun_d15 = [],[],[],[],[],[],[]

mon16,tue16,wed16,thu16,fri16,sat16,sun16 = [],[],[],[],[],[],[]
mon_d16,tue_d16,wed_d16,thu_d16,fri_d16,sat_d16,sun_d16 = [],[],[],[],[],[],[]


for i in range(len(days_num)):
    tmp = days_num[i]
    if tmp==1:
        mon.append(arr_del[i])
        mon_d.append(dep_del[i])
    elif tmp==2:
        tue.append(arr_del[i])
        tue_d.append(dep_del[i])
    elif tmp==3:
        wed.append(arr_del[i])
        wed_d.append(dep_del[i])
    elif tmp==4:
        thu.append(arr_del[i])
        thu_d.append(dep_del[i])
    elif tmp==5:
        fri.append(arr_del[i])
        fri_d.append(dep_del[i])
    elif tmp==6:
        sat.append(arr_del[i])
        sat_d.append(dep_del[i])
    elif tmp==7:
        sun.append(arr_del[i])
        sun_d.append(dep_del[i])



for i in range(len(days_num15)):
    tmp = days_num15[i]
    if tmp==1:
        mon15.append(arr_del15[i])
        mon_d15.append(dep_del15[i])
    elif tmp==2:
        tue15.append(arr_del15[i])
        tue_d15.append(dep_del15[i])
    elif tmp==3:
        wed15.append(arr_del15[i])
        wed_d15.append(dep_del15[i])
    elif tmp==4:
        thu15.append(arr_del15[i])
        thu_d15.append(dep_del15[i])
    elif tmp==5:
        fri15.append(arr_del15[i])
        fri_d15.append(dep_del15[i])
    elif tmp==6:
        sat15.append(arr_del15[i])
        sat_d15.append(dep_del15[i])
    elif tmp==7:
        sun15.append(arr_del15[i])
        sun_d15.append(dep_del15[i])



for i in range(len(days_num16)):
    tmp = days_num16[i]
    if tmp==1:
        mon16.append(arr16[i])
        mon_d16.append(dep16[i])
    elif tmp==2:
        tue16.append(arr16[i])
        tue_d16.append(dep16[i])
    elif tmp==3:
        wed16.append(arr16[i])
        wed_d16.append(dep16[i])
    elif tmp==4:
        thu16.append(arr16[i])
        thu_d16.append(dep16[i])
    elif tmp==5:
        fri16.append(arr16[i])
        fri_d16.append(dep16[i])
    elif tmp==6:
        sat16.append(arr16[i])
        sat_d16.append(dep16[i])
    elif tmp==7:
        sun16.append(arr16[i])
        sun_d16.append(dep16[i])




# CALCULATE MEAN AND MEDIAN FOR DAYS
# arrivals
#2014
mon_mean,mon_med = round(np.mean(mon),2),np.median(mon)
tue_mean,tue_med = round(np.mean(tue),2),np.median(tue)
wed_mean,wed_med = round(np.mean(wed),2),np.median(wed)
thu_mean,thu_med = round(np.mean(thu),2),np.median(thu)
fri_mean,fri_med = round(np.mean(fri),2),np.median(fri)
sat_mean,sat_med = round(np.mean(sat),2),np.median(sat)
sun_mean,sun_med = round(np.mean(sun),2),np.median(sun)

day_means = [mon_mean,tue_mean,wed_mean,thu_mean,fri_mean,sat_mean,sun_mean]
day_med = [mon_med,tue_med,wed_med,thu_med,fri_med,sat_med,sun_med]
# 2015
mon_mean15,mon_med15 = round(np.mean(mon15),2),np.median(mon15)
tue_mean15,tue_med15 = round(np.mean(tue15),2),np.median(tue15)
wed_mean15,wed_med15 = round(np.mean(wed15),2),np.median(wed15)
thu_mean15,thu_med15 = round(np.mean(thu15),2),np.median(thu15)
fri_mean15,fri_med15 = round(np.mean(fri15),2),np.median(fri15)
sat_mean15,sat_med15 = round(np.mean(sat15),2),np.median(sat15)
sun_mean15,sun_med15 = round(np.mean(sun15),2),np.median(sun15)
day_means15 = [mon_mean15,tue_mean15,wed_mean15,thu_mean15,fri_mean15,sat_mean15,sun_mean15]
day_med15 = [mon_med15,tue_med15,wed_med15,thu_med15,fri_med15,sat_med15,sun_med15]

# 2016
mon_mean16,mon_med16 = round(np.mean(mon16),2),np.median(mon16)
tue_mean16,tue_med16 = round(np.mean(tue16),2),np.median(tue16)
wed_mean16,wed_med16 = round(np.mean(wed16),2),np.median(wed16)
thu_mean16,thu_med16 = round(np.mean(thu16),2),np.median(thu16)
fri_mean16,fri_med16 = round(np.mean(fri16),2),np.median(fri16)
sat_mean16,sat_med16 = round(np.mean(sat16),2),np.median(sat16)
sun_mean16,sun_med16 = round(np.mean(sun16),2),np.median(sun16)
day_means16 = [mon_mean16,tue_mean16,wed_mean16,thu_mean16,fri_mean16,sat_mean16,sun_mean16]
day_med16 = [mon_med16,tue_med16,wed_med16,thu_med16,fri_med16,sat_med16,sun_med16]

# departures
#2014
mond_mean,mond_med = round(np.mean(mon_d),2),np.median(mon_d)
tued_mean,tued_med = round(np.mean(tue_d),2),np.median(tue_d)
wedd_mean,wedd_med = round(np.mean(wed_d),2),np.median(wed_d)
thud_mean,thud_med = round(np.mean(thu_d),2),np.median(thu_d)
frid_mean,frid_med = round(np.mean(fri_d),2),np.median(fri_d)
satd_mean,satd_med = round(np.mean(sat_d),2),np.median(sat_d)
sund_mean,sund_med = round(np.mean(sun_d),2),np.median(sun_d)

days_d_means = [mond_mean,tued_mean,wedd_mean,thud_mean,frid_mean,satd_mean,sund_mean]
days_d_med = [mond_med,tued_med,wedd_med,thud_med,frid_med,satd_med,sund_med]
#2015
mond_mean15,mond_med15 = round(np.mean(mon_d15),2),np.median(mon_d15)
tued_mean15,tued_med15 = round(np.mean(tue_d15),2),np.median(tue_d15)
wedd_mean15,wedd_med15 = round(np.mean(wed_d15),2),np.median(wed_d15)
thud_mean15,thud_med15 = round(np.mean(thu_d15),2),np.median(thu_d15)
frid_mean15,frid_med15 = round(np.mean(fri_d15),2),np.median(fri_d15)
satd_mean15,satd_med15 = round(np.mean(sat_d15),2),np.median(sat_d15)
sund_mean15,sund_med15 = round(np.mean(sun_d15),2),np.median(sun_d15)
days_d_means15 = [mond_mean15,tued_mean15,wedd_mean15,thud_mean15,frid_mean15,satd_mean15,sund_mean15]
days_d_med15 = [mond_med15,tued_med15,wedd_med15,thud_med15,frid_med15,satd_med15,sund_med15]

# 2016
mond_mean16,mond_med16 = round(np.mean(mon_d16),2),np.median(mon_d16)
tued_mean16,tued_med16 = round(np.mean(tue_d16),2),np.median(tue_d16)
wedd_mean16,wedd_med16 = round(np.mean(wed_d16),2),np.median(wed_d16)
thud_mean16,thud_med16 = round(np.mean(thu_d16),2),np.median(thu_d16)
frid_mean16,frid_med16 = round(np.mean(fri_d16),2),np.median(fri_d16)
satd_mean16,satd_med16 = round(np.mean(sat_d16),2),np.median(sat_d16)
sund_mean16,sund_med16 = round(np.mean(sun_d16),2),np.median(sun_d16)
days_d_means16 = [mond_mean16,tued_mean16,wedd_mean16,thud_mean16,frid_mean16,satd_mean16,sund_mean16]
days_d_med16 = [mond_med16,tued_med16,wedd_med16,thud_med16,frid_med16,satd_med16,sund_med16]


# PLOT DATA USING PYPLOT
plt.plot(range(1,len(week_mean_arr)+1),week_mean_arr,color='red',marker='o',label='Mean delay (arrivals  2014)')
plt.plot(range(1,len(week_med_arr)+1),week_med_arr,color='red',marker='o',linestyle='--',label='Median value (arrivals 2014)')
plt.plot(range(1,len(week_mean_arr15)+1),week_mean_arr15,color='green',marker='o',label='Mean delay (arrivals 2015)')
plt.plot(range(1,len(week_med_arr15)+1),week_med_arr15,color='green',marker='o',linestyle='--',label='Median value (arrivals 2015)')
plt.plot(range(1,len(week_mean_arr16)+1),week_mean_arr16,color='black',marker='o',label='Mean delay (arrivals 2016)')
plt.plot(range(1,len(week_med_arr16)+1),week_med_arr16,color='black',marker='o',linestyle='--',label='Median value (arrivals 2016)')
plt.title("Delays in weeks arrival (2014,2015,2016)")
plt.xlabel("Week")
plt.ylabel("Values in minutes")
plt.grid()
plt.xticks(np.arange(1,53,1))
plt.legend(loc='best')
plt.tight_layout()
plt.show()

plt.plot(range(1,len(week_mean_dep)+1),week_mean_dep,color='blue',marker='o',label='Mean delay (departure 2014)')
plt.plot(range(1,len(week_med_dep)+1),week_med_dep,color='blue',marker='o',linestyle='--',label='Median value (departure 2014)')
plt.plot(range(1,len(week_mean_dep15)+1),week_mean_dep15,color='purple',marker='o',label='Mean delay (departure 2015)')
plt.plot(range(1,len(week_med_dep15)+1),week_med_dep15,color='purple',marker='o',linestyle='--',label='Median value (departure 2015)')
plt.plot(range(1,len(week_mean_dep16)+1),week_mean_dep16,color='gray',marker='o',label='Mean delay (departure 2016)')
plt.plot(range(1,len(week_med_dep16)+1),week_med_dep16,color='gray',marker='o',linestyle='--',label='Median value (departure 2016)')

plt.title("Delays in weeks departure (2014,2015,2016)")
plt.xlabel("Week")
plt.ylabel("Values in minutes")
plt.grid()
plt.xticks(np.arange(1,53,1))
plt.legend(loc='best')
plt.tight_layout()
plt.show()







# arrivals median,mean plot
plot1 = plt.figure(1)
plt.plot(range(1,len(day_means)+1),day_means,color='red',marker='o',label='Mean value (arrival 2014)')
plt.plot(range(1,len(day_means15)+1),day_means15,color='blue',marker='o',label='Mean value (arrival 2015)')
plt.plot(range(1,len(day_means16)+1),day_means16,color='green',marker='o',label='Mean value (arrival 2016)')
plt.plot(range(1,len(day_med)+1),day_med,linestyle='--',color='red',marker='o',label='Median value (arrival 2014)')
plt.plot(range(1,len(day_med15)+1),day_med15,linestyle='--',color='blue',marker='o',label='Median value (arrival 2015)')
plt.plot(range(1,len(day_med16)+1),day_med16,linestyle='--',color='green',marker='o',label='Median value (arrival 2015)')
plt.grid()
plt.xticks(np.arange(1,8,1))
plt.xlabel("Days in week (Monday-->1. . .)")
plt.ylabel("Delays in minutes")
plt.legend(loc='best')
plt.title("Delays median by days (arrival)")

# departure median,mean plot
plot2 = plt.figure(2)
plt.plot(range(1,len(days_d_means)+1),days_d_means,color='red',marker='o',label='Mean value (departure 2014)')
plt.plot(range(1,len(days_d_means15)+1),days_d_means15,color='blue',marker='o',label='Mean value (departure 2015)')
plt.plot(range(1,len(days_d_means16)+1),days_d_means16,color='green',marker='o',label='Mean value (departure 2016)')
plt.plot(range(1,len(days_d_med)+1),days_d_med,linestyle='--',color='red',marker='o',label='Median value (departure 2014)')
plt.plot(range(1,len(days_d_med15)+1),days_d_med15,linestyle='--',color='blue',marker='o',label='Median value (departure 2015)')
plt.plot(range(1,len(days_d_med16)+1),days_d_med16,linestyle='--',color='green',marker='o',label='Median value (departure 2015)')
plt.xlabel("Days in week (Monday-->1. . .)")
plt.ylabel("Delays in minutes")
plt.grid()
plt.xticks(np.arange(1,8,1))
plt.legend(loc='best')
plt.title("Delays median by days (departure)")

plt.tight_layout()
plt.show()

del dates,dates15,dates16

carr = data2014['OP_CARRIER'].values
carr15 = data2015['OP_CARRIER'].values
carr16 = data2016['OP_CARRIER'].values

# getting all possible carrier info
mods = np.unique(carr)
mods15 = np.unique(carr15)
mods16 = np.unique(carr16)

# for 2014
mean_carr_arr=[]
mean_carr_dep=[]
med_carr_arr=[]
med_carr_dep=[]

#for 2015
mean_carr_arr15=[]
mean_carr_dep15=[]
med_carr_arr15=[]
med_carr_dep15=[]

#for 2016
mean_carr_arr16=[]
mean_carr_dep16=[]
med_carr_arr16=[]
med_carr_dep16=[]

#this is for 2014 data
for i in range(len(mods)):
    tmp = mods[i]
    tmp_arr=[]
    tmp_dep=[]
    for j in range(len(carr)):
        if tmp==carr[j]:
            tmp_arr.append(arr_del[j])
            tmp_dep.append(dep_del[j])
        
    mean_carr_arr.append(round(np.mean(tmp_arr),2))
    mean_carr_dep.append(round(np.mean(tmp_dep),2))
    med_carr_arr.append(round(np.median(tmp_arr),2))
    med_carr_dep.append(round(np.median(tmp_dep),2))

# this is for 2015 data
for i in range(len(mods15)):
    tmp = mods15[i]
    tmp_arr=[]
    tmp_dep=[]
    for j in range(len(carr15)):
        if tmp==carr15[j]:
            tmp_arr.append(arr_del15[j])
            tmp_dep.append(dep_del15[j])
        
    mean_carr_arr15.append(round(np.mean(tmp_arr),2))
    mean_carr_dep15.append(round(np.mean(tmp_dep),2))
    med_carr_arr15.append(round(np.median(tmp_arr),2))
    med_carr_dep15.append(round(np.median(tmp_dep),2))


# this is for 2016 data
for i in range(len(mods16)):
    tmp = mods16[i]
    tmp_arr=[]
    tmp_dep=[]
    for j in range(len(carr16)):
        if tmp==carr16[j]:
            tmp_arr.append(arr16[j])
            tmp_dep.append(dep16[j])
        
    mean_carr_arr16.append(round(np.mean(tmp_arr),2))
    mean_carr_dep16.append(round(np.mean(tmp_dep),2))
    med_carr_arr16.append(round(np.median(tmp_arr),2))
    med_carr_dep16.append(round(np.median(tmp_dep),2))

del carr,carr15,carr16

fig,ax = plt.subplots(3,1,figsize=(10,6))

# ploting 2014 data delay by carrier
ax[0].plot(range(1,len(mods)+1),mean_carr_arr,color='red',marker='o',label='Mean value (arrival)')
ax[0].plot(range(1,len(mods)+1),mean_carr_dep,color='blue',marker='o',label='Mean value (departure)')
ax[0].plot(range(1,len(mods)+1),med_carr_arr,color='red',marker='o',linestyle='--',label='Median value (arrival)')
ax[0].plot(range(1,len(mods)+1),med_carr_dep,color='blue',marker='o',linestyle='--',label='Median value (departure)')
ax[0].set_title("Mean and median delays by carrier (2014)")
ax[0].set_ylabel("Values in minutes")
ax[0].grid()
ax[0].set_xticks(np.arange(1,len(mods)+1,1))
ax[0].set_xticklabels(mods,rotation='vertical',fontsize=10)

# ploting 2015 data delay by carrier
ax[1].plot(range(1,len(mods15)+1),mean_carr_arr15,color='red',marker='o',label='Mean value (arrival)')
ax[1].plot(range(1,len(mods15)+1),mean_carr_dep15,color='blue',marker='o',label='Mean value (departure)')
ax[1].plot(range(1,len(mods15)+1),med_carr_arr15,color='red',marker='o',linestyle='--',label='Median value (arrival)')
ax[1].plot(range(1,len(mods15)+1),med_carr_dep15,color='blue',marker='o',linestyle='--',label='Median value (departure)')
ax[1].set_title("Mean and median delays by carrier (2015)")
ax[1].set_ylabel("Values in minutes")
ax[1].grid()
ax[1].set_xticks(np.arange(1,len(mods15)+1,1))
ax[1].set_xticklabels(mods15,rotation='vertical',fontsize=10)

#ploting 2016 data delay by carrier
ax[2].plot(range(1,len(mods16)+1),mean_carr_arr16,color='red',marker='o',label='Mean value (arrival)')
ax[2].plot(range(1,len(mods16)+1),mean_carr_dep16,color='blue',marker='o',label='Mean value (departure)')
ax[2].plot(range(1,len(mods16)+1),med_carr_arr16,color='red',marker='o',linestyle='--',label='Median value (arrival)')
ax[2].plot(range(1,len(mods16)+1),med_carr_dep16,color='blue',marker='o',linestyle='--',label='Median value (departure)')
ax[2].set_title("Mean and median delays by carrier (2016)")
ax[2].set_ylabel("Values in minutes")
ax[2].grid()
ax[2].set_xticks(np.arange(1,len(mods16)+1,1))
ax[2].set_xticklabels(mods16,rotation='vertical',fontsize=10)

plt.legend(loc='best')
plt.tight_layout()
plt.show()



# ORIGIN

orig = data2014['ORIGIN'].values
orig15 = data2015['ORIGIN'].values
orig16 = data2016['ORIGIN'].values

# FIND ALL VALUES IN 2014,2015,2016
mod,mod15,mod16 = np.unique(orig),np.unique(orig15),np.unique(orig16)


# for 2014
mean_orig_arr=[]
mean_orig_dep=[]
med_orig_arr=[]
med_orig_dep=[]

#for 2015
mean_orig_arr15=[]
mean_orig_dep15=[]
med_orig_arr15=[]
med_orig_dep15=[]

#for 2016
mean_orig_arr16=[]
mean_orig_dep16=[]
med_orig_arr16=[]
med_orig_dep16=[]

#this is for 2014 data
for i in range(len(mod)):
    tmp = mod[i]
    tmp_arr=[]
    tmp_dep=[]
    for j in range(len(orig)):
        if tmp==orig[j]:
            tmp_arr.append(arr_del[j])
            tmp_dep.append(dep_del[j])
        
    mean_orig_arr.append(round(np.mean(tmp_arr),2))
    mean_orig_dep.append(round(np.mean(tmp_dep),2))
    med_orig_arr.append(round(np.median(tmp_arr),2))
    med_orig_dep.append(round(np.median(tmp_dep),2))

# this is for 2015 data
for i in range(len(mod15)):
    tmp = mod15[i]
    tmp_arr=[]
    tmp_dep=[]
    for j in range(len(orig15)):
        if tmp==orig15[j]:
            tmp_arr.append(arr_del15[j])
            tmp_dep.append(dep_del15[j])
        
    mean_orig_arr15.append(round(np.mean(tmp_arr),2))
    mean_orig_dep15.append(round(np.mean(tmp_dep),2))
    med_orig_arr15.append(round(np.median(tmp_arr),2))
    med_orig_dep15.append(round(np.median(tmp_dep),2))


# this is for 2016 data
for i in range(len(mod16)):
    tmp = mod16[i]
    tmp_arr=[]
    tmp_dep=[]
    for j in range(len(orig16)):
        if tmp==orig16[j]:
            tmp_arr.append(arr16[j])
            tmp_dep.append(dep16[j])
        
    mean_orig_arr16.append(round(np.mean(tmp_arr),2))
    mean_orig_dep16.append(round(np.mean(tmp_dep),2))
    med_orig_arr16.append(round(np.median(tmp_arr),2))
    med_orig_dep16.append(round(np.median(tmp_dep),2))

del orig,orig15,orig16

fig,ax = plt.subplots(3,1,figsize=(10,6))

# ploting 2014 data delay by origin
ax[0].plot(range(1,len(mod)+1),mean_orig_arr,color='red',marker='o',label='Mean value (arrival)')
ax[0].plot(range(1,len(mod)+1),mean_orig_dep,color='blue',marker='o',label='Mean value (departure)')
ax[0].plot(range(1,len(mod)+1),med_orig_arr,color='red',marker='o',linestyle='--',label='Median value (arrival)')
ax[0].plot(range(1,len(mod)+1),med_orig_dep,color='blue',marker='o',linestyle='--',label='Median value (departure)')
ax[0].set_title("Mean and median delays by origin (2014)")
ax[0].set_ylabel("Values in minutes")
ax[0].grid()
ax[0].set_xticks(np.arange(1,len(mod)+1,1))
ax[0].set_xticklabels(mod,rotation='vertical',fontsize=6)

# ploting 2015 data delay by origin
ax[1].plot(range(1,len(mod15)+1),mean_orig_arr15,color='red',marker='o',label='Mean value (arrival)')
ax[1].plot(range(1,len(mod15)+1),mean_orig_dep15,color='blue',marker='o',label='Mean value (departure)')
ax[1].plot(range(1,len(mod15)+1),med_orig_arr15,color='red',marker='o',linestyle='--',label='Median value (arrival)')
ax[1].plot(range(1,len(mod15)+1),med_orig_dep15,color='blue',marker='o',linestyle='--',label='Median value (departure)')
ax[1].set_title("Mean and median delays by origin (2015)")
ax[1].set_ylabel("Values in minutes")
ax[1].grid()
ax[1].set_xticks(np.arange(1,len(mod15)+1,1))
ax[1].set_xticklabels(mod15,rotation='vertical',fontsize=6)

#ploting 2016 data delay by origin 
ax[2].plot(range(1,len(mod16)+1),mean_orig_arr16,color='red',marker='o',label='Mean value (arrival)')
ax[2].plot(range(1,len(mod16)+1),mean_orig_dep16,color='blue',marker='o',label='Mean value (departure)')
ax[2].plot(range(1,len(mod16)+1),med_orig_arr16,color='red',marker='o',linestyle='--',label='Median value (arrival)')
ax[2].plot(range(1,len(mod16)+1),med_orig_dep16,color='blue',marker='o',linestyle='--',label='Median value (departure)')
ax[2].set_title("Mean and median delays by origin (2016)")
ax[2].set_ylabel("Values in minutes")
ax[2].grid()
ax[2].set_xticks(np.arange(1,len(mod16)+1,1))
ax[2].set_xticklabels(mod16,rotation='vertical',fontsize=6)

plt.legend(loc='best')
plt.tight_layout()
plt.show()



# DESTINATION

dest = data2014['DEST'].values
dest15 = data2015['DEST'].values
dest16 = data2016['DEST'].values

mod,mod15,mod16 = np.unique(dest),np.unique(dest15),np.unique(dest16)


# for 2014
mean_dest_arr=[]
mean_dest_dep=[]
med_dest_arr=[]
med_dest_dep=[]

#for 2015
mean_dest_arr15=[]
mean_dest_dep15=[]
med_dest_arr15=[]
med_dest_dep15=[]

#for 2016
mean_dest_arr16=[]
mean_dest_dep16=[]
med_dest_arr16=[]
med_dest_dep16=[]

#this is for 2014 data
for i in range(len(mod)):
    tmp = mod[i]
    tmp_arr=[]
    tmp_dep=[]
    for j in range(len(dest)):
        if tmp==dest[j]:
            tmp_arr.append(arr_del[j])
            tmp_dep.append(dep_del[j])
        
    mean_dest_arr.append(round(np.mean(tmp_arr),2))
    mean_dest_dep.append(round(np.mean(tmp_dep),2))
    med_dest_arr.append(round(np.median(tmp_arr),2))
    med_dest_dep.append(round(np.median(tmp_dep),2))

# this is for 2015 data
for i in range(len(mod15)):
    tmp = mod15[i]
    tmp_arr=[]
    tmp_dep=[]
    for j in range(len(dest15)):
        if tmp==dest15[j]:
            tmp_arr.append(arr_del15[j])
            tmp_dep.append(dep_del15[j])
        
    mean_dest_arr15.append(round(np.mean(tmp_arr),2))
    mean_dest_dep15.append(round(np.mean(tmp_dep),2))
    med_dest_arr15.append(round(np.median(tmp_arr),2))
    med_dest_dep15.append(round(np.median(tmp_dep),2))


# this is for 2016 data
for i in range(len(mod16)):
    tmp = mod16[i]
    tmp_arr=[]
    tmp_dep=[]
    for j in range(len(dest16)):
        if tmp==dest16[j]:
            tmp_arr.append(arr16[j])
            tmp_dep.append(dep16[j])
        
    mean_dest_arr16.append(round(np.mean(tmp_arr),2))
    mean_dest_dep16.append(round(np.mean(tmp_dep),2))
    med_dest_arr16.append(round(np.median(tmp_arr),2))
    med_dest_dep16.append(round(np.median(tmp_dep),2))

del dest,dest15,dest16

fig,ax = plt.subplots(3,1,figsize=(10,6))

# ploting 2014 data delay by destination
ax[0].plot(range(1,len(mod)+1),mean_dest_arr,color='red',marker='o',label='Mean value (arrival)')
ax[0].plot(range(1,len(mod)+1),mean_dest_dep,color='blue',marker='o',label='Mean value (departure)')
ax[0].plot(range(1,len(mod)+1),med_dest_arr,color='red',marker='o',linestyle='--',label='Median value (arrival)')
ax[0].plot(range(1,len(mod)+1),med_dest_dep,color='blue',marker='o',linestyle='--',label='Median value (departure)')
ax[0].set_title("Mean and median delays by destination (2014)")
ax[0].set_ylabel("Values in minutes")
ax[0].grid()
ax[0].set_xticks(np.arange(1,len(mod)+1,1))
ax[0].set_xticklabels(mod,rotation='vertical',fontsize=6)

# ploting 2015 data delay by destination
ax[1].plot(range(1,len(mod15)+1),mean_dest_arr15,color='red',marker='o',label='Mean value (arrival)')
ax[1].plot(range(1,len(mod15)+1),mean_dest_dep15,color='blue',marker='o',label='Mean value (departure)')
ax[1].plot(range(1,len(mod15)+1),med_dest_arr15,color='red',marker='o',linestyle='--',label='Median value (arrival)')
ax[1].plot(range(1,len(mod15)+1),med_dest_dep15,color='blue',marker='o',linestyle='--',label='Median value (departure)')
ax[1].set_title("Mean and median delays by destination (2015)")
ax[1].set_ylabel("Values in minutes")
ax[1].grid()
ax[1].set_xticks(np.arange(1,len(mod15)+1,1))
ax[1].set_xticklabels(mod15,rotation='vertical',fontsize=6)

#ploting 2016 data delay by destinatio
ax[2].plot(range(1,len(mod16)+1),mean_dest_arr16,color='red',marker='o',label='Mean value (arrival)')
ax[2].plot(range(1,len(mod16)+1),mean_dest_dep16,color='blue',marker='o',label='Mean value (departure)')
ax[2].plot(range(1,len(mod16)+1),med_dest_arr16,color='red',marker='o',linestyle='--',label='Median value (arrival)')
ax[2].plot(range(1,len(mod16)+1),med_dest_dep16,color='blue',marker='o',linestyle='--',label='Median value (departure)')
ax[2].set_title("Mean and median delays by destination (2016)")
ax[2].set_ylabel("Values in minutes")
ax[2].grid()
ax[2].set_xticks(np.arange(1,len(mod16)+1,1))
ax[2].set_xticklabels(mod16,rotation='vertical',fontsize=6)

plt.legend(loc='best')
plt.tight_layout()
plt.show()
