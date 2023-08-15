from data_load_pre import MyDataset
from resnet_3 import model_resnet

from predict_ostime import predict

import os
import torch
import torch.utils.data
import torch.nn as nn
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

def train_net(net, train_txt, save_train_dict, lr=0.001, batch_size=50, epochs=50):
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    net.to(device=device)

    if not os.path.exists(save_train_dict):
        os.makedirs(save_train_dict)
    # 将网络拷贝到deivce中
    train_dataset = MyDataset(train_txt)
    train_dataLoader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                                shuffle=True,
                                                num_workers=2,
                                                prefetch_factor=2,
                                                pin_memory=True
                                                )
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-10, weight_decay=0.05, amsgrad=True)
    scheduler = CosineAnnealingWarmRestarts(optimizer,T_0=5,T_mult=2) #3 22/46 5 36/76

    # 定义Loss算法
    criterion_BCE = nn.BCELoss()
    criterion_BCE_pre = nn.BCELoss()
    criterion_BCE_doc = nn.BCELoss()

    for epoch in range(0, epochs):
    # train
        net.train()
        best_loss = 100000000000000000000000.0
        temp_total = 0
        train_loss = 0

        with tqdm(total=len(train_dataset), desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for ct, pet, doctor_glod, data_list  in train_dataLoader:
            
                ct = ct.to(device=device, dtype=torch.float32)
                pet = pet.to(device=device, dtype=torch.float32)
                doctor_glod = doctor_glod.to(device=device, dtype=torch.float32)
                data_list = data_list.to(device=device, dtype=torch.float32)
                
                ct_pet = torch.cat((ct, pet), dim=1)
                
                out, out_c, pre_os, doc_os = net(ct_pet, doctor_glod)
                
                if epoch>3:
                    for batch in range(ct_pet.shape[0]):
                        if out_c[batch][0] != doctor_glod[batch][0] and abs(pre_os[batch][0]-data_list[batch][1]) < abs(doc_os[batch][0]-data_list[batch][1]):
                            doctor_glod[batch][0] = out_c[batch][0]
                
                loss_pre = criterion_BCE_pre(pre_os, data_list[:,1:2])
                loss_doc = criterion_BCE_doc(doc_os, data_list[:,1:2])
                loss_up = criterion_BCE(out, doctor_glod)
                
                if epoch<3:
                    loss = 0.01*loss_up + (loss_pre + loss_doc)
                else:
                    loss = loss_up + 0.3*(loss_pre + loss_doc)
                
                train_loss+=loss.item()
                
                # 保存loss值最小的网络参数
                if loss < best_loss:
                    best_loss = loss
                    torch.save(net.state_dict(), save_train_dict + '/' + str(epoch + 1) + '.pth')
                    # 更新参数
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                pbar.update(pet.shape[0])
                temp_total = temp_total + 1
                pbar.set_postfix({'loss': '{0:1.3f}'.format(train_loss/temp_total), 
                                  })
            scheduler.step()
            
 
if __name__ == "__main__":
    # 指定训练集地址，开始训练
    net=model_resnet()
    print('2_112_216')
    print('!!!Train!!!')
    print("##################################################################")
    train_txt = './order/train_2_216_112.txt'
    save_train_dict = './params/2_216_112'
    train_net(net, train_txt, save_train_dict)

    print('!!!Predict!!!')
    model_dir_all = './params/2_216_112'
    train_txt = './order/test_2_216_112.txt'
    flag = predict(model_dir_all, train_txt)
  
