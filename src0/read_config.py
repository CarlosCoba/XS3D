from csv import reader
import csv

def config_file(name):
	list_dict = []
	with open(name, 'r') as file:
		#csv_file = csv.DictReader(file, delimiter = "\t")
		csv_file = csv.DictReader(filter(lambda row: row[0]!='#', file), delimiter = "\t")

		for row in csv_file:
			list_dict.append(dict(row))

	return list_dict


