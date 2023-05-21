"""docstring"""
import csv
import matplotlib.pyplot as plt

with open('stage_2_train_labels.csv', 'r', encoding = "utf-8") as csv_file:
    reader = csv.reader(csv_file)
    index, count0, count1, areaSum = 0, 0, 0, 0
    data = []
    for row in reader:
        if index == 0:
            index += 1
            continue
        if row[5] == '1':
            count1 += 1
        else:
            count0 += 1
        if(row[3] != '' and row[4] != ''):
            area = float(row[3]) * float(row[4])
            areaSum += area
            data.append(area)

    print("Instances of pneumonia: " + str(count1))
    print("No instances of pneumonia: " + str(count0))
    mean = areaSum / count1
    print("Average size of bounding box: " + str(mean))
    csv_file.close()

plt.hist(data, bins = 20)
plt.axvline(mean, color = 'r', linestyle = 'dashed', linewidth = 1)
plt.xlabel('Area of Bounding Box')
plt.ylabel('Frequency')
plt.show()
