import os
data_dir = '/home/vince/datasets/nuscenes_kitti_exported'

folder_path = os.path.join(data_dir,"label_2") # replace with the path to your folder

# get a list of all files in the folder
files = os.listdir(folder_path)

# loop through each file and rename it to a number starting from 0
for i, file in enumerate(files):
    name = file.split('.')[0]
    new_name = str(i)
    # deleted_file=True
    # with open(os.path.join(data_dir,"label_2",name+".txt")) as f:
    #     for line in f:
    #         words = line.split()
    #         if words[0] == "car":
    #             deleted_file=False
    # print(deleted_file)
    # if deleted_file:
    #     print("delete file")
    #     os.unlink(os.path.join(data_dir,"image_2",name+".png"))
    #     os.unlink(os.path.join(data_dir,"label_2",name+".txt"))
    #     os.unlink(os.path.join(data_dir,"calib",name+".txt"))
    # else:
    os.rename(os.path.join(data_dir,"image_2",name+".png"), os.path.join(data_dir,"image_2",new_name+".png"))
    os.rename(os.path.join(data_dir,"label_2",name+".txt"), os.path.join(data_dir,"label_2",new_name+".txt"))
    os.rename(os.path.join(data_dir,"calib",name+".txt"), os.path.join(data_dir,"calib",new_name+".txt"))