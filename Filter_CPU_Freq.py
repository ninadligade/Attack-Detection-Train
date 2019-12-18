import sys
import os

clamp = lambda n, minn, maxn: max(min(maxn, n), minn)

filepath = sys.argv[1]
column_number=int(sys.argv[2])
upper_bound=int(sys.argv[3])
lower_bound=int(sys.argv[4])
new_file = open(filepath+"_new","w+")
newlines = []

with open(filepath) as fp:
   line = fp.readline()
   while line:
       l=list(map(int,line.split(",")))
       l[column_number] = clamp(int(l[column_number]), lower_bound, upper_bound)
       new_file.write(str(l).strip('[]').replace(" ", "")+"\n")
       line = fp.readline()

os.replace(filepath, filepath+"_old")
os.replace(filepath+"_new", filepath)