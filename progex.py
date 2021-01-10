
L0 = 1000 # number of cells healthy
L1 = [0]
L2 = [] # day_steps at this level
L3 = [] # day_steps at this level
L4 = [] # day_steps at this level
L5 = 0 # number of cells dead


for day_step in range_of_steps:  # chunks of 20 days
	number_of_probable_transfers = f(P = Ae(kt))

	L0 -= number_of_probable_transfers
	
	L1.insert(0,number_of_probable_transfers)
	if len(L1)>3:
		next_level = L1.pop(3) 

	L2.insert(0,next_level)
	
	if len(L2)>3:
		next_level = L2.pop(3) 
		L3.insert(0,next_level)

	if len(L3)>3:
		next_level = L3.pop(3) 
		L4.insert(0,next_level)

	if len(L4)>3:
		next_level = L4.pop(3) 
		L5 += next_level
		
	# Nengo: the number of neurons in each ensemble / level
	# weighted average of time neurons have been in that level
		
	