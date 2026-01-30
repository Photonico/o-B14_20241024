import os

path = os.getcwd()
print(path)
DIR=[x[0] for x in os.walk(path)]
print(DIR[0])
inumjobs = len(DIR)
print('The number of sub directories is '+str(inumjobs))
submit_command = "qsub run_cpu.csh"

for i in range(0, inumjobs):
  if(DIR[i]!=path):
    os.chdir(DIR[i])
    print(DIR[i])
    os.system(submit_command)
    os.chdir('../')
    print(os.getcwd())
  
print("Yeah! All done!")
