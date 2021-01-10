import math
import pandas as pd
import numpy as np

#This is a synapse loss function DESCRIBE MORE
#For the data file, in each value, the first value of the list is the proportion of cells in the circuit affected by that group and the second is the synaptic transmission for each of them


#Variables
neurons = 1000 #Total number of neurons in the system
day_step_size = 5 #How often do you want to simulate the model in days
tot_days = 2000 #Total number of days you want to run for - here 0 - 1000 inclusive
total_steps = tot_days / day_step_size

#Function to calculate the synapse transmission as time continues on 
def synapse_trans(tau, t):
	trans = math.exp((math.log(0.9)*t)/tau)
	if trans <= (1/math.e):
		trans = 0.0
	return trans

#This is the initial probability function that DOES NOT account for the previous number affected
def prob(t):
	probability_aff = (0.004 + (0.000001*t))
	return probability_aff

prev_num_affected = 0
day_and_num_affected = [] #List of lists that have [day number, number affected by amyloid or dead] 
#This loop goes through each time in the total days and finds the number of cells affected by amyloid plaques and appends the day and number to the end of the list above
for t in range(0, tot_days + 1): #Goes from 0 to 1000
	if t == 0:
		num_affected = prob(t)*neurons
	else:
		num_affected = prev_num_affected + (prev_num_affected * prob(t))
	if num_affected < 0:
		num_affected = 0
	prev_num_affected = num_affected
	if t % day_step_size == 0 or t == 0:
		day_and_num_affected.append([t, num_affected])

#This calculates the new number of cells that will be affected after the current day step
list_new_affected = [] #List of total cells that will be affected after this day - index is the day step (Day 0 is 0, Day 20 is 1, etc)
prev_days_total = 4.0
for x in day_and_num_affected:
	new_affected = x[-1] - prev_days_total

	prev_days_total = x[-1]
	list_new_affected.append(new_affected)

#This creates a dataframe that holds all the data needed for the simulation

init_day = day_and_num_affected[0]
L0 = [neurons - init_day[1]] #Total Healthy cells
L1 = [0, 0, 0, 0, 0] #Total cells with mild amyloid #25 days in this group
L2 = [0, 0, 0, 0, 0, 0] #Total cells with medium amyloid #30 days in this group
L3 = [0, 0, 0, 0, 0, 0, 0] #Total cells with high amyloid #35 days in this group
L4 = [0, 0, 0, 0, 0, 0, 0, 0] #Total cells with severe amyloid #40 days in this group
L5 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] #Total cells with very severe amyloid #55 days in this group
L6 = [0] #Total cells with no connections - dead

#Synapse transmission for each category based on index based on calculations above
L0_synapse_function = [1.0]
L1_synapse_function = [0.9740037464, 0.9486832981, 0.9240210865, 0.9, 0.8766033718]
L2_synapse_function = [0.8538149682, 0.8316189778, 0.81, 0.7889430346, 0.7684334714, 0.74845708]
L3_synapse_function = [0.729, 0.7100487311, 0.6915901243, 0.673611372, 0.6561, 0.639043858, 0.6224311119]
L4_synapse_function = [0.6062502348, 0.59049, 0.5751394722, 0.5601880007, 0.5456252114, 0.531441, 0.517625525, 0.5041692006]
L5_synapse_function = [0.4910626902, 0.4782969, 0.4658629725, 0.4537522805, 0.4419564212, 0.43046721, 0.4192766753, 0.4083770525, 0.3977607791, 0.387420489, 0.3773490077]
L6_synapse_function = [0.0]



final_list = []
for step in range(0, total_steps + 1): #Here, the step size is 5
	day = step*day_step_size

	L1_avg = np.ma.average(L1_synapse_function, axis=0, weights=L1)
	L2_avg = np.ma.average(L2_synapse_function, axis=0, weights=L2)
	L3_avg = np.ma.average(L3_synapse_function, axis=0, weights=L3)
	L4_avg = np.ma.average(L4_synapse_function, weights=L4)
	L5_avg = np.ma.average(L5_synapse_function, axis=0, weights=L5)

	L0_prop = sum(L0)/neurons
	L1_prop = sum(L1)/neurons
	L2_prop = sum(L2)/neurons
	L3_prop = sum(L3)/neurons
	L4_prop = sum(L4)/neurons
	L5_prop = sum(L5)/neurons
	L6_prop = sum(L6)/neurons

	if L0_prop < 0:
		L0_prop = 0
	if L1_prop < 0:
		L1_prop = 0
	if L2_prop < 0: 
		L2_prop = 0
	if L3_prop < 0:
		L3_prop = 0
	if L4_prop < 0:
		L4_prop = 0
	if L5_prop < 0:
		L5_prop = 0
	if L6_prop < 0:
		L6_prop = 0


	final_list.append([day, [L0_prop, 1.0], [L1_prop, L1_avg], [L2_prop, L2_avg], [L3_prop, L3_avg], [L4_prop, L4_avg], [L5_prop, L5_avg], [L6_prop, 0.0]])

	#Move everything over by one
	number_transfers = list_new_affected[step]
	if (L0[0] - number_transfers) < 0:
		L0[0] = 0
		L1.insert(0, 0)
	else:
		L0[0] -= number_transfers
		L1.insert(0, number_transfers)
	if len(L1) > 5:
		next_level = L1.pop(-1)
		L2.insert(0, next_level)

	if len(L2) > 6:
		next_level = L2.pop(-1)
		L3.insert(0, next_level)

	if len(L3) > 7:
		next_level = L3.pop(-1)
		L4.insert(0, next_level)

	if len(L4) > 8:
		next_level = L4.pop(-1)
		L5.insert(0, next_level)

	if len(L5) > 11:
		next_level = L5.pop(-1)
		num = next_level
		L6[0] += num
		print(L0, L1)
	print(L0, L1)


data_final = pd.DataFrame(final_list)
data_final.columns = ["Day Number", "L0", "L1", "L2", "L3", "L4", "L5", "L6"]
data_final.to_csv('synapse_loss_data.csv', index=False)

print data_final.head()

#Create dataframe that holds synapse transmission from day 0 to 200 - After day 185, cell loses connections
#This is for tau = 20 days - amount of time for 10% loss in the synapse
df = []
for t in range(0,201,5):
	df.append([t, synapse_trans(20, t)])

data = pd.DataFrame(df)
data.columns = ["Day Number", "Amount of Synapse Transmission"]

