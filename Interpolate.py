import os


def interpolate(service_data):
    file = open(service_data)
    output_file = open("Data/temp.txt", "w")
    lines = file.readlines()
    output_file.write(lines[0])
    for i in range(1, len(lines)-1):
        first = lines[i].split(",")
        second = lines[i+1].split(",")
        diff = int(second[0]) - int(first[0])
        for j in range(diff):
            temp = lines[i].split(",")
            temp[0] = str(int(first[0]) + j)
            output_file.write(','.join(temp))
    output_file.write(lines[len(lines)-1])
    os.rename(service_data, service_data+"_old")
    os.rename("Data/temp.txt", service_data)


interpolate("Data/service_data.txt")