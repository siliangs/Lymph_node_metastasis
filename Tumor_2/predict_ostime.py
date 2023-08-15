from data_load_predict import MyDataset
from resnet_3 import model_resnet

import numpy as np
import torch
from sklearn.metrics import confusion_matrix,roc_auc_score


def calculate_metric(gt, pred): 
    # pred[pred>0.5]=1
    # pred[pred<1]=0
    confusion = confusion_matrix(gt,pred)
    TP = confusion[1, 1]
    TN = confusion[0, 0]
    FP = confusion[0, 1]
    FN = confusion[1, 0]
    print('Accuracy:',(TP+TN)/float(TP+TN+FP+FN))
    print('Sensitivity:',TP / float(TP+FN))
    print('Specificity:',TN / float(TN+FP)) 
    print('PPV:',TP / float(TP + FP))
    print('NPV:',TN / float(TN + FN))
    

def predict(model_dir_all, train_txt):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 加载网络
    net = model_resnet()
    net.to(device=device)

    
    # 将网络拷贝到deivce中
    train_dataset = MyDataset(train_txt)
    train_dataLoader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=100,
                                                shuffle=True,
                                                num_workers=2,
                                                prefetch_factor=2,
                                                pin_memory=True
                                                )
    # 加载模型参数
    dir_name = '50.pth'
    doctor_right_1 = 0
    doctor_right_0 = 0
    net_right_1 = 0
    net_right_0 = 0
    out_all = []
    out_5_all = []
    label_all = []
    doctor_all = []
    patient_all = []
    model_dir = model_dir_all + '/' + dir_name
    # print(model_dir)
    net.load_state_dict(torch.load(model_dir, map_location='cuda'))
    # 测试模式
    net.eval()
    with torch.no_grad():
        for ct, pet, doctor_glod, glod, patient in train_dataLoader:

            ct = ct.to(device=device, dtype=torch.float32)
            pet = pet.to(device=device, dtype=torch.float32)
            doctor_glod = doctor_glod.to(device=device, dtype=torch.float32)

            ct_pet = torch.cat((ct, pet), dim=1)

            out, out_c, pre_os, doc_os = net(ct_pet, doctor_glod)

            out_auc = np.array(out.data.cpu())
            out = np.array(out.data.cpu())
            out[out>0.5]=1
            out[out<=0.5]=0

            for bacth in range(out.shape[0]):
                if int(doctor_glod[bacth][0]) == int(glod[bacth][0]) == 1:
                    doctor_right_1+=1
                if int(doctor_glod[bacth][0]) == int(glod[bacth][0]) == 0:
                    doctor_right_0+=1

                if int(out[bacth][0]) == int(glod[bacth][0]) == 1:
                    net_right_1+=1
                if int(out[bacth][0]) == int(glod[bacth][0]) == 0:
                    net_right_0+=1

                out_all.append(out_auc[bacth][0])
                out_5_all.append(int(out[bacth][0]))
                label_all.append(int(glod[bacth][0]))
                doctor_all.append(int(doctor_glod[bacth][0]))
                patient_all.append(int(patient[bacth][0]))
        print(dir_name)
        print('out_all=', out_all)
        print('out_5_all=', out_5_all)
        print('label_all=', label_all)
        print('doctor_all=', doctor_all)
        print('patient_all=', patient_all)
        print('pre:')
        AUC_value = roc_auc_score(label_all,out_all)
        calculate_metric(label_all, out_5_all)
        print('AUC_value:', AUC_value)
        print('\n')
        print('doctor_glod:')
        calculate_metric(label_all, doctor_all)
        print('\n')
        
        print('doctor_right:', doctor_right_1+doctor_right_0)
        print('doctor_right_1:', doctor_right_1)
        print('doctor_right_0:', doctor_right_0)
        print('net_right:', net_right_1+net_right_0)
        print('net_right_1:', net_right_1)
        print('net_right_0:', net_right_0)
        print('\n')

if __name__ == "__main__":
    model_dir_all = './params/2_216_112'
    train_txt = './order/test_2_216_112.txt'

    predict(model_dir_all, train_txt)




