import csv
import matplotlib.pyplot as plt

with open('/home/ec2-user/rsna/stage_2_train_labels.csv', 'r', encoding = "utf-8") as csv_file:
    reader = csv.reader(csv_file)
    index, count_0, count_1, area_Sum, data = 0, 0, 0, 0, []
    for row in reader:
        if index == 0:
            index += 1
            continue
        if row[5] == '1':
            count_1 += 1
        else:
            count_0 += 1
        if(row[3] != '' and row[4] != ''):
            area = float(row[3]) * float(row[4])
            area_Sum += area
            data.append(area)
    print("Instances of pneumonia: " + str(count_1))
    print("No instances of pneumonia: " + str(count_0))
    print("Average size of bounding box: " + str(area_Sum / count_1))
    csv_file.close()
plt.hist(data, bins = 20)
plt.axvline(area_Sum / count_1, color = 'r', linestyle = 'dashed', linewidth = 1)
plt.xlabel('Area of Bounding Box')
plt.ylabel('Frequency')
plt.show()
