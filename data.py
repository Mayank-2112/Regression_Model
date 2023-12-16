
# importing the csv module
import csv
import random





        
# field names
fields = ['Start_time','End_time','Service_time','Queue Length']
 
# data rows of csv file
rows=[]
for i in range(300):
    st=random.randint(10,50)
    e=random.randint(20,70)
    a=random.randint(1,30)
    if st<e:
        s=e-st
    rows.append([st,e,s,a])
# name of csv file
filename = "data.csv"
 
# writing to csv file
with open(filename, 'w') as csvfile:
    # creating a csv writer object
    csvwriter = csv.writer(csvfile)
     
    # writing the fields
    csvwriter.writerow(fields)
     
    # writing the data rows
    csvwriter.writerows(rows)
    print('record entered')