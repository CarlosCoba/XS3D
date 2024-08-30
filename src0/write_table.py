import numpy as np
from astropy.table import Table, Column, MaskedColumn
from astropy.io import ascii
import csv



"""
def write(data,name):
	#a+: Open the file for reading and writing and creates new file if it doesn’t exist. All additions are made at the end of the file and no existing data can be modified.
	#a: Open the file for writing and creates new file if it doesn’t exist. All additions are made at the end of the file and no existing data can be modified.
	#
	data = [data]
	with open(name, 'a+') as csvfile:

		writer = csv.writer(csvfile)
		[writer.writerow(r) for r in data]


"""



def write(data,name,column = True):
	#a+: Open the file for reading and writing and creates new file if it doesn’t exist. All additions are made at the end of the file and no existing data can be modified.
	#a: Open the file for writing and creates new file if it doesn’t exist. All additions are made at the end of the file and no existing data can be modified.
	#

	
	#To write in columns
	if column == True:
		data_col = [[i] for i in data]
		with open(name, 'a') as csvfile:

			writer = csv.writer(csvfile)
			writer.writerows(data_col)

	#To write in rows
	else:
		data = [data]
		with open(name, 'a') as csvfile:

			writer = csv.writer(csvfile)
			[writer.writerow(r) for r in data]


