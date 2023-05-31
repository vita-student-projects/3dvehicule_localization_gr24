import os
data_dir = '/home/vince/datasets/nuscenes_kitti_exported'



# get a list of all files in the folder
final = []
file_text  =os.path.join(data_dir,"ImageSets","test.txt")
with open(file_text, 'r') as file:
    for line2 in file:
        file = os.path.join(data_dir,"testing","label_2",line2.strip()+".txt")
        deleted=True
        with open(file,'r') as file:
            for line in file:
                
                first_word = line.strip().split()[0]
                # print(first_word == "car",first_word)
                if first_word == "car":
                    deleted = False
        if deleted == False:
            final.append(line2.strip())
with open(file_text, 'w') as file:
    for item in final:
        file.write(item + '\n')