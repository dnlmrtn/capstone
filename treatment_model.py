#Heres a basic model that Serdar was recommending
anxLvl = 0.7    #Score 0-1 on anxiety level based on hamilton score
dose = 3 #dose in mg
anxChg = -dose*0.1
anxNew = anxLvl + anxChg
