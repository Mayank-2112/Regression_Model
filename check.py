import random
start_time=[]
end_time=[]
for i in range(300):
    a=random.randint(10,50)
    start_time.append(a)


for i in range(300):
    a=random.randint(20,70)
    end_time.append(a)

    
service_time=[]
for i in range(300):
    if start_time[i]<end_time[i]:
        a=end_time[i]-start_time[i]
        service_time.append(a)
print(service_time)
        