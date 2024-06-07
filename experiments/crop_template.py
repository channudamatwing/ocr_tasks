import os
import shutil
import cv2
import pandas as pd
from khmerocr.preprocessing import crop_image, read_image

# print("-----Start initilize output directory-----")
# save_path = "outputs"
# if not os.path.exists(save_path):
#     os.makedirs(save_path)
# else:
#     shutil.rmtree(save_path)
#     os.makedirs(save_path)
# print("-----Output directory is completely initilized-----")


# print("-----Start to crop images-----")
# image_path = "generated_templates_new"
# if os.path.exists(image_path):
#     files = os.listdir(image_path)
#     for file in files:
#         save_name = str(int(file.split(".")[0].split("-")[1]) - 1) + ".png"
#         image = read_image(f"{image_path}/{file}")
#         cropped_image = crop_image(image, kernal_size=(8,8))
#         cv2.imwrite(f"{save_path}/{save_name}", cropped_image)
#     print("-----Images are completely cropped-----")

print("-----Start to read data from excel-----")
df = pd.read_excel("templates.xlsx")
df.rename(columns={"ID": "0", 
                   "រាជធានី/ក្រុង": "1",
                   "ក្រុង/ស្រុក/ខណ្ឌ": "2",
                   "ឃុំ/សង្កាត់": "3",
                   "ភូមិ": "4",
                   "សន្លឹកផែនទី": "5",
                   "លេខក្បាលដី": "6",
                   "ទំហំ": "7",
                   "ប្រភេទដី": "8",
                   "ប្រើប្រាស់": "9",
                   "លក្ខណៈ": "10"}, inplace=True)
print("-----Dataframe is completely initilized-----")

print("-----Start to initilize output directory-----")
save_path = "cleaned_outputs_(new)"
if not os.path.exists(save_path):
    os.makedirs(save_path)
else:
    shutil.rmtree(save_path)
    os.makedirs(save_path)
print("-----Output directory is completely initilized-----")

print("-----Start to crop images-----")
coor_list = [(207,129,76,22), # 0 
             (100,155,76,22), # 1
             (100,177,76,22), # 2
             (100,199,76,22), # 3
             (100,221,76,22), # 4
             (355,158,76,22), # 5
             (355,175,76,22), # 6
             (355,196,76,22), # 7
             (355,218,76,22), # 8
             (360,242,76,22), # 9
             (387,264,72,22)] # 10
image_path = "outputs"
if os.path.exists(image_path):
    files = os.listdir(image_path)
    counter = 0
    labels = []
    for file in files:
        image = read_image(f"{image_path}/{file}")
        image = cv2.resize(image, (467, 687), interpolation = cv2.INTER_LINEAR)
        idx = int(file.split(".")[0])
        column = 0
        for coor in coor_list:
            cropped_image = crop_image(image, points=coor)
            labels.append(str(df.iloc[idx][str(column)]))
            save_name = "img_" + str(counter) + ".png"
            cv2.imwrite(f"{save_path}/{save_name}", cropped_image)
            column = column + 1
            counter = counter + 1
    pd.DataFrame({'label': labels}).to_csv(f"{save_path}/labels.csv", index=False)
