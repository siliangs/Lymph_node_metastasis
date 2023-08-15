import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import SimpleITK as sitk


def get_data_list(txt_flie):
    path = '../data_328'
    txt_112 = '../excel/glod_112.xlsx'
    txt_216 = '../excel/glod_216.xlsx'
    tumor_num_path = '../excel/Tumor_num.xlsx'
    ostime_path = '../excel/2019_last_contact.xlsx'
    
    data_list=[]
    txt_112_flie = pd.read_excel(txt_112)
    txt_216_flie = pd.read_excel(txt_216)
    tumor_num_flie = pd.read_excel(tumor_num_path)
    ostime_flie = pd.read_excel(ostime_path)
    
    ID_112 = np.array(txt_112_flie['ID'].dropna(how='all'))

    for slic in range(len(txt_flie)):
        patient=str(txt_flie.values[slic][0])
        patient_name=patient.split('_')
        
        if len(patient_name)==1:
            patient_name_in=patient
        elif len(patient_name)==2:
            patient_name_in=patient_name[0]
        
        if int(patient_name_in) in ID_112:
            index_glod = txt_112_flie[txt_112_flie.ID == int(patient_name_in)]
        else:
            index_glod = txt_216_flie[txt_216_flie.ID == int(patient_name_in)]
        
        num = index_glod.index.tolist()[0]
        doctor_glod = index_glod.loc[num, 'doctor_glod']
        
        patient_ostime_axis = ostime_flie[ostime_flie.ID == int(patient_name_in)]
        osmonth = patient_ostime_axis['osmonth'].tolist()[0]
        if osmonth>60:
            osmonth = 1.0
        else:
            osmonth = 0.0
        
        patient_tumor_num= tumor_num_flie[tumor_num_flie.ID == int(patient_name_in)]
        tumor_num = patient_tumor_num['Tumor_num'].tolist()[0]
            
        CT_path = path+'/'+patient+'/CT.nii'
        PET_path = path+'/'+patient+'/PET.nii'

        temp_dict = {'CT': CT_path, 'PET': PET_path, 'doctor_glod':doctor_glod,'osmonth':osmonth,
                      'tumor_num':tumor_num}
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
        osmonth=self.path_dict[index]['osmonth']
        tumor_num=self.path_dict[index]['tumor_num']
        
        CT_img = sitk.GetArrayFromImage(sitk.ReadImage(CT_path))
        PET_img = sitk.GetArrayFromImage(sitk.ReadImage(PET_path))

        CT_img = (CT_img - np.min(CT_img)) / (np.max(CT_img) - np.min(CT_img))

        CT_img = CT_img[np.newaxis, :, :]
        PET_img = PET_img[np.newaxis, :, :]

        doctor_glod_np = np.array([doctor_glod])
        
        data_list = np.array([tumor_num, osmonth])
        
        
        return torch.from_numpy(CT_img), torch.from_numpy(PET_img), torch.from_numpy(
            doctor_glod_np), torch.from_numpy(data_list)
    
if __name__ == '__main__':
    data=MyDataset('./order/train_2_216_112.txt')
    print(data[0][0].shape)
    print(data[0][1].shape)
    print(data[0][2].shape)
    
