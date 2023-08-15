import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import SimpleITK as sitk


def get_data_list(txt_flie):
    path = '../data_328'
    txt_112 = '../excel/glod_112.xlsx'
    data_list=[]
    txt_112_flie = pd.read_excel(txt_112)
    
    for slic in range(len(txt_flie)):
        patient=str(txt_flie.values[slic][0])
        index_glod = txt_112_flie[txt_112_flie.ID == int(patient)]

        num = index_glod.index.tolist()[0]
        doctor_glod = index_glod.loc[num, 'doctor_glod']
        glod = index_glod.loc[num, 'glod']
        
        CT_path = path+'/'+patient+'/CT.nii'
        PET_path = path+'/'+patient+'/PET.nii'
        
        temp_dict = {'CT': CT_path, 'PET': PET_path, 'doctor_glod':doctor_glod, 'glod':glod, 'patient':int(patient)}
        data_list.append(temp_dict)
    return data_list

class MyDataset(Dataset):
    def __init__(self,txt_path):
        self.txt = pd.read_table(txt_path)
        self.path_dict = get_data_list(self.txt)
        
    def __len__(self):
        return len(self.txt)

    def __getitem__(self, index):
        CT_path=self.path_dict[index]['CT']
        PET_path=self.path_dict[index]['PET']
        doctor_glod=self.path_dict[index]['doctor_glod']
        glod=self.path_dict[index]['glod']
        patient=self.path_dict[index]['patient']
        
        CT_img = sitk.GetArrayFromImage(sitk.ReadImage(CT_path))
        PET_img = sitk.GetArrayFromImage(sitk.ReadImage(PET_path))

        CT_img = (CT_img - np.min(CT_img)) / (np.max(CT_img) - np.min(CT_img))

        CT_img = CT_img[np.newaxis, :, :]
        PET_img = PET_img[np.newaxis, :, :]

        doctor_glod_np = np.array([doctor_glod])
        glod_np = np.array([glod])
        patient_np = np.array([patient])
        
        return torch.from_numpy(CT_img), torch.from_numpy(PET_img), torch.from_numpy(doctor_glod_np), torch.from_numpy(glod_np), torch.from_numpy(patient_np)
    
if __name__ == '__main__':
    data=MyDataset('./order/test_2_216_112.txt')
    print(data[0][0].shape)
    print(data[0][1].shape)
    print(data[0][2].shape)
    # print(data[0][0])
    # print(data[0][1])
    # print(data[0][6].shape)
    # print(data[0][4])
    
