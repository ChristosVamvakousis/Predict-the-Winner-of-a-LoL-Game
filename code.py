import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier, MLPRegressor
import warnings
from sklearn.exceptions import ConvergenceWarning
import json

warnings.filterwarnings(action='ignore', category=ConvergenceWarning)
print('Starting..')
allData = pd.read_csv('games.csv')
firstModel = allData[['t1_champ1id','t1_champ2id','t1_champ3id','t1_champ4id','t1_champ5id','t2_champ1id','t2_champ2id','t2_champ3id','t2_champ4id','t2_champ5id','winner']]
firstModel = firstModel.drop_duplicates()

earlygameModel = allData[['t1_champ1id','t1_champ2id','t1_champ3id','t1_champ4id','t1_champ5id','t2_champ1id','t2_champ2id','t2_champ3id','t2_champ4id','t2_champ5id','winner','firstBlood','firstTower']]
earlygameModel = earlygameModel.drop_duplicates()

lategameModel = allData[['t1_champ1id','t1_champ2id','t1_champ3id','t1_champ4id','t1_champ5id','t2_champ1id','t2_champ2id','t2_champ3id','t2_champ4id','t2_champ5id','winner','firstBlood','firstTower','firstDragon','firstBaron','firstInhibitor']]
lategameModel = lategameModel.drop_duplicates()
#firstModel = firstModel.to_csv('Cleaned-data.csv', index=False)
#data = pd.read_csv('Cleaned-data.csv')
################################################################################################################################################################## GOING INTO TRAINING


######################################################	firstModel
print('Going into training for firstModel')
Xfirst = firstModel.drop('winner', axis=1) # select all columns except target
yfirst = firstModel['winner'] # select target variable
X_trainfirst, X_testfirst, y_trainfirst, y_testfirst = train_test_split(Xfirst, yfirst, test_size=0.2, random_state=42) 
scalerFirst = StandardScaler()
X_trainfirst = scalerFirst.fit_transform(X_trainfirst)
X_testfirst = scalerFirst.transform(X_testfirst)
first = GradientBoostingClassifier()
first.fit(X_trainfirst, y_trainfirst)
scores_firstModel = cross_val_score(GradientBoostingClassifier(), X_trainfirst, y_trainfirst, cv=5)
print('firstModel have Accuracy: ' + str(np.mean(scores_firstModel)))


######################################################	earlygameModel
print('Going into training for earlygameModel') 
Xearly = earlygameModel.drop('winner', axis=1) # select all columns except target
yearly = earlygameModel['winner'] # select target variable
X_trainearly, X_testearly, y_trainearly, y_testearly = train_test_split(Xearly, yearly, test_size=0.2, random_state=42) 
scalerEarly = StandardScaler()
X_trainearly = scalerEarly.fit_transform(X_trainearly)
X_testearly = scalerEarly.transform(X_testearly)
early = GradientBoostingClassifier()
early.fit(X_trainearly , y_trainearly)
scores_earlygameModel = cross_val_score(GradientBoostingClassifier(), X_trainearly, y_trainearly, cv=5)
print('earlygameModel have Accuracy: ' + str(np.mean(scores_earlygameModel)))


######################################################	lategameModel


print('Going into training for lategameModel') 
Xlate = lategameModel.drop('winner', axis=1) # select all columns except target
ylate = lategameModel['winner'] # select target variable
X_trainlate, X_testlate, y_trainlate, y_testlate = train_test_split(Xlate, ylate, test_size=0.2, random_state=42) 
scalerLate = StandardScaler()
X_trainlate = scalerLate.fit_transform(X_trainlate)
X_testlate = scalerLate.transform(X_testlate)
late = GradientBoostingClassifier()
late.fit(X_trainlate , y_trainlate)
scores_lategameModel = cross_val_score(GradientBoostingClassifier(), X_trainlate, y_trainlate, cv=5)
print('lategameModel have Accuracy: ' + str(np.mean(scores_lategameModel)))

################################################################################################################################################################## GOING INTO VALUE COLLECTING
with open('champion_info.json') as f:
	data = json.load(f)
team1 = []
team2 = []

print("\nGive the champions that ", end='') #"Give the champions that PURPLE Team have picked"
print("\033[35m" + "PURPLE", end='')
print("\033[0m"+" Team have picked!")

for i in range(5):
	flag = False
	while flag == False: #Until the user inputs a true value
		x = input('player' +str(i+1) + ': ')
		x = x.upper()
		for champ_id in data['data']: #Checking value
			champion = data['data'][champ_id] 
			if champion['name'].upper() == x or champion['key'].upper() == x:
				if champion['id'] not in team1:
					team1.append(champion['id'])
					flag = True
		if flag == False: #if we have reached the end of json file
			print('\033[91m' + 'Can not find that champion or champion is already in use, try again!'+ '\033[0m') #prints in red and resets
	
print("\nGive the champions that ", end='') #"Give the champions that BLUE Team have picked"
print("\033[34m" + "BLUE", end='')
print("\033[0m"+" Team have picked!")

for i in range(5):
	flag = False
	while flag == False: #Until the user inputs a true value
		x = input('player' +str(i+1) + ': ')
		x = x.upper()
		for champ_id in data['data']: #Checking value
			champion = data['data'][champ_id] 
			if champion['name'].upper() == x or champion['key'].upper() == x:
				if champion['id'] not in team1:
					if champion['id'] not in team2:
						team2.append(champion['id'])
						flag = True
		if flag == False: #if we have reached the end of json file
			print('\033[91m' + 'Can not find that champion or champion is already in use, try again!'+ '\033[0m') #prints in red and resets

################################################################################################################################################################## PASTED READING PHASE
print('\n')
team1 = np.array(team1)
team2 = np.array(team2)
firstApproach = np.concatenate((team1, team2)).reshape(1, 10)
datatoTest = pd.DataFrame(firstApproach, columns=['t1_champ1id','t1_champ2id','t1_champ3id','t1_champ4id','t1_champ5id','t2_champ1id','t2_champ2id','t2_champ3id','t2_champ4id','t2_champ5id'])

new_data_scaled = scalerFirst.transform(datatoTest)
predictions = first.predict(new_data_scaled)
if predictions == 1:
    print('Its more likely team ' + "\033[35m" + 'PURPLE' + "\033[0m" + ' to win!')
else:
    print('Its more likely team ' + "\033[34m" + 'BLUE' + "\033[0m" + ' to win!')
##################################### phase 2
answer = 3
print('\nGoing into Phase 2, which team got the first blood?')
print('1. for team '+ "\033[35m" + 'PURPLE' + "\033[0m")
print('2. for team '+ "\033[34m" + 'BLUE' + "\033[0m")
answer = int(input())
while answer != 1 and answer != 2:
	print('\033[91m' + 'Error, try again'+ '\033[0m')
	answer = int(input())
earlyApproach = np.concatenate((firstApproach, [[answer]]), axis=1)


answer = 3
print('Which team got the first tower?')
print('1. for team '+ "\033[35m" + 'PURPLE' + "\033[0m")
print('2. for team '+ "\033[34m" + 'BLUE' + "\033[0m")
answer = int(input())
while answer != 1 and answer != 2:
	print('\033[91m' + 'Error, try again'+ '\033[0m')
	answer = int(input())
earlyApproach = np.concatenate((earlyApproach, [[answer]]), axis=1)


dataEarly = pd.DataFrame(earlyApproach, columns=['t1_champ1id','t1_champ2id','t1_champ3id','t1_champ4id','t1_champ5id','t2_champ1id','t2_champ2id','t2_champ3id','t2_champ4id','t2_champ5id','firstBlood','firstTower'])
dataEarly_scaled = scalerEarly.transform(dataEarly)
predictions = early.predict(dataEarly_scaled)
if predictions == 1:
    print('Its more likely team ' + "\033[35m" + 'PURPLE' + "\033[0m" + ' to win!')
else:
    print('Its more likely team ' + "\033[34m" + 'BLUE' + "\033[0m" + ' to win!')

##################################### phase 3

answer = 3
print('\nGoing into Phase 3, which team got the first dragon?')
print('1. for team '+ "\033[35m" + 'PURPLE' + "\033[0m")
print('2. for team '+ "\033[34m" + 'BLUE' + "\033[0m")
answer = int(input())
while answer != 1 and answer != 2:
	print('\033[91m' + 'Error, try again'+ '\033[0m')
	answer = int(input())
lateApproach = np.concatenate((earlyApproach, [[answer]]), axis=1)


answer = 3
print('Which team got the first Baron?')
print('1. for team '+ "\033[35m" + 'PURPLE' + "\033[0m")
print('2. for team '+ "\033[34m" + 'BLUE' + "\033[0m")
answer = int(input())
while answer != 1 and answer != 2:
	print('\033[91m' + 'Error, try again'+ '\033[0m')
	answer = int(input())
lateApproach = np.concatenate((lateApproach, [[answer]]), axis=1)


answer = 3
print('Which team got the first Inhibitor?')
print('1. for team '+ "\033[35m" + 'PURPLE' + "\033[0m")
print('2. for team '+ "\033[34m" + 'BLUE' + "\033[0m")
answer = int(input())
while answer != 1 and answer != 2:
	print('\033[91m' + 'Error, try again'+ '\033[0m')
	answer = int(input())
lateApproach = np.concatenate((lateApproach, [[answer]]), axis=1)




dataLate = pd.DataFrame(lateApproach, columns=['t1_champ1id','t1_champ2id','t1_champ3id','t1_champ4id','t1_champ5id','t2_champ1id','t2_champ2id','t2_champ3id','t2_champ4id','t2_champ5id','firstBlood','firstTower','firstDragon','firstBaron','firstInhibitor'])
dataLate_scaled = scalerLate.transform(dataLate)
predictions = late.predict(dataLate_scaled)
if predictions == 1:
    print('Its more likely team ' + "\033[35m" + 'PURPLE' + "\033[0m" + ' to win!')
else:
    print('Its more likely team ' + "\033[34m" + 'BLUE' + "\033[0m" + ' to win!')


