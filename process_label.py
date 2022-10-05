import pandas as pd
from tqdm import tqdm
import random
import numpy as np


data = pd.read_csv("Data_Entry_2017_v2020.csv")
data = data.rename(columns={"Image Index": "Image", "Finding Labels": "Labels", "Patient ID": "Patient_id"}) 

label_dictionary = {'Cardiomegaly': 0, 'Emphysema': 1, 'Effusion': 2, 'Hernia': 3, 'Infiltration': 4, 'Mass': 5, 'Nodule': 6, 'Atelectasis': 7, 'Pneumothorax': 8, 'Pleural_Thickening': 9, 'Pneumonia': 10, 'Fibrosis': 11, 'Edema': 12, 'Consolidation': 13}
print(label_dictionary)
all_label_ids = []
for multilabel in data.Labels:
    label_list = list(multilabel.split("|"))
    current_label_ids=""
    for label in label_list:
        if label != "No Finding":
            current_label_ids += str(label_dictionary[label]) +","
    all_label_ids.append(current_label_ids)
data['label_ids'] = all_label_ids
print(data.head())


test_list = pd.read_csv("test_list.txt",header=None).rename(columns={0: "Image"})
train_val_list = pd.read_csv("train_val_list.txt",header=None).rename(columns={0: "Image"})

def validation_split_patient(train_val_list, data_entry):
    train_list=[]
    val_list=[]

    print(len(train_val_list))
    print(train_val_list.head())
    num_validation_images = len(train_val_list)//8
    total_patitents = len(data_entry.Patient_id.unique())
    # approximate number of patients in validation set
    num_val_patients = total_patitents//8
    random.seed(3)
    val_patient_ids = random.sample(range(0, total_patitents), num_val_patients)
    
    for sample in tqdm(range(len(train_val_list))):
        current_patient_id = list(data_entry[data_entry["Image"] == train_val_list.Image[sample]].Patient_id)
        if current_patient_id[0] in val_patient_ids:
            val_list.append(train_val_list.Image[sample])
        else:
            train_list.append(train_val_list.Image[sample])

    train_list = pd.DataFrame(train_list,columns=['Image'])
    val_list = pd.DataFrame(val_list,columns=['Image'])
    return train_list,val_list

total_samples = len(test_list) + len(train_val_list)
print("total",total_samples)
train_set, val_set = validation_split_patient(train_val_list,data)
# print("train", len(train_set)/total_samples)
# print("val", len(val_set)/total_samples)
# print("test", len(test_list)/total_samples)


print(label_dictionary)

def string_to_N_hot(string: str):
    if string== "No Finding":
        return np.zeros((14,), dtype=float)
    true_index = [label_dictionary[cl] for cl in string.split("|")]
    label = np.zeros((14,), dtype=float)
    label[true_index] = 1
    return label

data["labels"] = data["Labels"].apply(string_to_N_hot)
data[["Image Index", "labels"]].rename(columns={"Image Index": "file_name"}).to_json("images/metadata.jsonl", orient='records', lines=True)


def assign_img_dir(file: str):
    return "/mnt/bd/medai-cv/cvpr/CXR8/images/images/"+file

def map_df(Img: str):
    label = data[data["Image"]==Img].labels.item()
    return label
test_list['labels'] = test_list['Image'].apply(map_df)
test_list['Image'] = test_list['Image'].apply(assign_img_dir)
test_list.to_json("images/test.jsonl", orient='records', lines=True)

print(test_list.head())

train_set['labels'] = train_set['Image'].apply(map_df)
train_set['Image'] = train_set['Image'].apply(assign_img_dir)
train_set.to_json("images/train.jsonl", orient='records', lines=True)

val_set['labels'] = val_set['Image'].apply(map_df)
val_set['Image'] = val_set['Image'].apply(assign_img_dir)
val_set.to_json("images/val.jsonl", orient='records', lines=True)