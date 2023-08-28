
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import random
from matplotlib import cm
import spectral as spy
from sklearn import metrics
import time
from sklearn import preprocessing
import torch
import LDA_SLIC
import offset_graph

from thop import profile
import os
from scipy.fftpack import fft,ifft
from ulits import Draw_Classification_Map,get_Samples_GT,GT_To_One_Hot







device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

samples_type = ['ratio', 'same_num'][1] 

for (FLAG, curr_train_ratio,Scale) in [(3,30,200)]: 

    torch.cuda.empty_cache()
    OA_ALL = []
    AA_ALL = []
    KPP_ALL = []
    AVG_ALL = []

    Train_Time_ALL=[]    
    Test_Time_ALL=[]
   
   
    Seed_List=[0]
    
    if FLAG == 1:
        data_mat = sio.loadmat('..\\HyperImage_data\\indian\\Indian_pines_corrected.mat')
        data = data_mat['indian_pines_corrected']
        gt_mat = sio.loadmat('..\\HyperImage_data\\indian\\Indian_pines_gt.mat')
        gt = gt_mat['indian_pines_gt']
       
        val_ratio = 0.05  
        class_count = 16 
        learning_rate = 5e-4  
        max_epoch =400 
        dataset_name = "indian_"  
        
        pass
    if FLAG == 2:
        data_mat = sio.loadmat('..\\HyperImage_data\\paviaU\\PaviaU.mat')
        data = data_mat['paviaU']
        gt_mat = sio.loadmat('..\\HyperImage_data\\paviaU\\Pavia_University_gt.mat')
        gt = gt_mat['paviaU_gt']
        
       
        val_ratio = 0.005  
        class_count = 9  
        learning_rate = 5e-4  
        max_epoch = 400 
        dataset_name = "paviaU_"  
        
        pass
    if FLAG == 3:
        data_mat = sio.loadmat('..\\HyperImage_data\\Salinas\\Salinas_corrected.mat')
        data = data_mat['salinas']
        gt_mat = sio.loadmat('..\\HyperImage_data\\Salinas\\Salinas_gt.mat')
        gt = gt_mat['salinas_gt']
        val_ratio = 0.05  
        class_count = 16  
        learning_rate = 5e-4  
        max_epoch = 400  
        dataset_name = "salinas_"  
       
        pass
    

    if samples_type == 'same_num': val_ratio = 6 
    superpixel_scale = Scale
    train_ratio = curr_train_ratio
    cmap = cm.get_cmap('jet', class_count + 1)
    plt.set_cmap(cmap)
    m, n, d = data.shape  

    orig_data=data
    height, width, bands = data.shape  
    data = np.reshape(data, [height * width, bands])
    minMax = preprocessing.StandardScaler()
    data = minMax.fit_transform(data)
    data = np.reshape(data, [height, width, bands])
    
    gt_reshape=np.reshape(gt, [-1])
    samplesCount_list = []
    for i in range(class_count):
        idx = np.where(gt_reshape == i + 1)[-1]
        samplesCount = len(idx)
        samplesCount_list.append(samplesCount)
    print(samplesCount_list)


    model_dir = './results/' + dataset_name + "/" 
    if not os.path.isdir(model_dir):
         os.makedirs(model_dir, exist_ok=True)
    
    Draw_Classification_Map(gt,model_dir+dataset_name) 

    for curr_seed in Seed_List:
        train_samples_gt, test_samples_gt, val_samples_gt= get_Samples_GT(curr_seed, gt, class_count, curr_train_ratio,val_ratio, samples_type )
        train_samples_gt_onehot = GT_To_One_Hot(train_samples_gt, class_count)
        test_samples_gt_onehot = GT_To_One_Hot(test_samples_gt, class_count)
        val_samples_gt_onehot = GT_To_One_Hot(val_samples_gt, class_count)
        train_samples_gt_onehot = np.reshape(train_samples_gt_onehot, [-1, class_count]).astype(int)
        test_samples_gt_onehot = np.reshape(test_samples_gt_onehot, [-1, class_count]).astype(int)
        val_samples_gt_onehot = np.reshape(val_samples_gt_onehot, [-1, class_count]).astype(int)
        Test_GT = np.reshape(test_samples_gt, [m, n])  
        
        
        train_val_test_gt=[train_samples_gt,val_samples_gt,test_samples_gt]
        for type in range(3):
            gt_reshape = np.reshape(train_val_test_gt[type], [-1])
            print("===============================")
            samplesCount_list=[]
            for i in range(class_count):
                idx = np.where(gt_reshape == i + 1)[-1]
                samplesCount = len(idx)
                samplesCount_list.append(samplesCount)
            print(samplesCount_list)
        
        train_label_mask = np.zeros([m * n, class_count])
        temp_ones = np.ones([class_count])
        train_samples_gt = np.reshape(train_samples_gt, [m * n])
        for i in range(m * n):
            if train_samples_gt[i] != 0:
                train_label_mask[i] = temp_ones
        train_label_mask = np.reshape(train_label_mask, [m* n, class_count])


        test_label_mask = np.zeros([m * n, class_count])
        temp_ones = np.ones([class_count])
        test_samples_gt = np.reshape(test_samples_gt, [m * n])
        for i in range(m * n):
            if test_samples_gt[i] != 0:
                test_label_mask[i] = temp_ones
        test_label_mask = np.reshape(test_label_mask, [m* n, class_count])


        val_label_mask = np.zeros([m * n, class_count])
        temp_ones = np.ones([class_count])
        val_samples_gt = np.reshape(val_samples_gt, [m * n])
        for i in range(m * n):
            if val_samples_gt[i] != 0:
                val_label_mask[i] = temp_ones
        val_label_mask = np.reshape(val_label_mask, [m* n, class_count])

       
                                                                
       
        ls = LDA_SLIC.LDA_SLIC(data, np.reshape( train_samples_gt,[height,width]), class_count-1)
        tic0=time.time()
        Q, S ,A,Seg= ls.simple_superpixel(scale=superpixel_scale)
        
        
        
        toc0 = time.time()
        
        Q=torch.from_numpy(Q).to(device)
        A=torch.from_numpy(A).to(device)

        train_samples_gt=torch.from_numpy(train_samples_gt.astype(np.float32)).to(device)
        test_samples_gt=torch.from_numpy(test_samples_gt.astype(np.float32)).to(device)
        val_samples_gt=torch.from_numpy(val_samples_gt.astype(np.float32)).to(device)

        train_samples_gt_onehot = torch.from_numpy(train_samples_gt_onehot.astype(np.float32)).to(device)
        test_samples_gt_onehot = torch.from_numpy(test_samples_gt_onehot.astype(np.float32)).to(device)
        val_samples_gt_onehot = torch.from_numpy(val_samples_gt_onehot.astype(np.float32)).to(device)

        train_label_mask = torch.from_numpy(train_label_mask.astype(np.float32)).to(device)
        test_label_mask = torch.from_numpy(test_label_mask.astype(np.float32)).to(device)
        val_label_mask = torch.from_numpy(val_label_mask.astype(np.float32)).to(device)

       
        
       
        height, width, bands = data.shape
        net_input=np.array( data,np.float32)

        net_input=torch.from_numpy(net_input.astype(np.float32)).to(device)
        

        
        if dataset_name == "indian_":
            net = offset_graph.offset_graph(height, width, bands, class_count, Q, A, model='smoothed')
        else:
            net = offset_graph.offset_graph(height, width, bands, class_count, Q, A)

        print("parameters", net.parameters(), len(list(net.parameters())))
        net.to(device)
        MACs, params = profile(net, inputs=(net_input,), verbose=False)

        print('MACs: %0.2f M, params: %0.2f K' % (MACs / 1e6, params / 1e3))

        
        def compute_loss(predict: torch.Tensor, reallabel_onehot: torch.Tensor, reallabel_mask: torch.Tensor):
            real_labels = reallabel_onehot
            we = -torch.mul(real_labels,torch.log(predict))
            we = torch.mul(we, reallabel_mask) 
            pool_cross_entropy = torch.sum(we)
            return pool_cross_entropy
        

        zeros = torch.zeros([m * n]).to(device).float()
        def evaluate_performance(network_output,train_samples_gt,train_samples_gt_onehot, require_AA_KPP=False,printFlag=True):
            if False==require_AA_KPP:
                with torch.no_grad():
                    available_label_idx=(train_samples_gt!=0).float()
                    available_label_count=available_label_idx.sum()
                    correct_prediction =torch.where(torch.argmax(network_output, 1) ==torch.argmax(train_samples_gt_onehot, 1),available_label_idx,zeros).sum()
                    OA= correct_prediction.cpu()/available_label_count
                    
                    return OA
            else:
                with torch.no_grad():
                    
                    available_label_idx=(train_samples_gt!=0).float()
                    available_label_count=available_label_idx.sum()
                    correct_prediction =torch.where(torch.argmax(network_output, 1) ==torch.argmax(train_samples_gt_onehot, 1),available_label_idx,zeros).sum()
                    OA= correct_prediction.cpu()/available_label_count
                    OA=OA.cpu().numpy()
                    
                    
                    zero_vector = np.zeros([class_count])
                    output_data=network_output.cpu().numpy()
                    train_samples_gt=train_samples_gt.cpu().numpy()
                    train_samples_gt_onehot=train_samples_gt_onehot.cpu().numpy()
                    
                    output_data = np.reshape(output_data, [m * n, class_count])
                    idx = np.argmax(output_data, axis=-1)
                    for z in range(output_data.shape[0]):
                        if ~(zero_vector == output_data[z]).all():
                            idx[z] += 1
                    
                    count_perclass = np.zeros([class_count])
                    correct_perclass = np.zeros([class_count])
                    for x in range(len(train_samples_gt)):
                        if train_samples_gt[x] != 0:
                            count_perclass[int(train_samples_gt[x] - 1)] += 1
                            if train_samples_gt[x] == idx[x]:
                                correct_perclass[int(train_samples_gt[x] - 1)] += 1
                    test_AC_list = correct_perclass / count_perclass
                    test_AA = np.average(test_AC_list)

                    
                    test_pre_label_list = []
                    test_real_label_list = []
                    output_data = np.reshape(output_data, [m * n, class_count])
                    idx = np.argmax(output_data, axis=-1)
                    idx = np.reshape(idx, [m, n])
                    for ii in range(m):
                        for jj in range(n):
                            if Test_GT[ii][jj] != 0:
                                test_pre_label_list.append(idx[ii][jj] + 1)
                                test_real_label_list.append(Test_GT[ii][jj])
                    test_pre_label_list = np.array(test_pre_label_list)
                    test_real_label_list = np.array(test_real_label_list)
                    kappa = metrics.cohen_kappa_score(test_pre_label_list.astype(np.int16),
                                                      test_real_label_list.astype(np.int16))
                    test_kpp = kappa

                    
                    if printFlag:
                        print("test OA=", OA, "AA=", test_AA, 'kpp=', test_kpp)
                        print('acc per class:')
                        print(test_AC_list)

                    OA_ALL.append(OA)
                    AA_ALL.append(test_AA)
                    KPP_ALL.append(test_kpp)
                    AVG_ALL.append(test_AC_list)
                      
                    
                    f = open(model_dir + dataset_name + '_results.txt', 'a+')
                    str_results = '\n======================' \
                                  + " learning rate=" + str(learning_rate) \
                                  + " epochs=" + str(max_epoch) \
                                  + " train ratio=" + str(train_ratio) \
                                  + " val ratio=" + str(val_ratio) \
                                  + " ======================" \
                                  + "\nOA=" + str(OA) \
                                  + "\nAA=" + str(test_AA) \
                                  + '\nkpp=' + str(test_kpp) \
                                  + '\nacc per class: ' + str(test_AC_list) \
                                 
                                  
                    f.write(str_results)
                    f.close()
                    return OA
        
        
        
        
        optimizer = torch.optim.Adam(net.parameters(),lr=learning_rate)# weight_decay=0.0001
        best_loss=99999
        best_OA=0
        Stop_Flag=0
        net.train()
        tic1 = time.perf_counter()
        
        for i in range(max_epoch+1):
            optimizer.zero_grad()  
            output= net(net_input)
            loss = compute_loss(output,train_samples_gt_onehot,train_label_mask)
            loss.backward(retain_graph=False)
            
            optimizer.step()  
            if i%10==0:
                with torch.no_grad():
                    net.eval()
                    output= net(net_input,showFlag=True)
                    
                    trainloss = compute_loss(output, train_samples_gt_onehot, train_label_mask)
                    trainOA = evaluate_performance(output, train_samples_gt, train_samples_gt_onehot)
                    valloss = compute_loss(output, val_samples_gt_onehot, val_label_mask)
                    valOA = evaluate_performance(output, val_samples_gt, val_samples_gt_onehot)
                    print("{}\ttrain loss={}\t train OA={} val loss={}\t val OA={}".format(str(i + 1), trainloss, trainOA, valloss, valOA))

                    if valloss < best_loss  or valOA>best_OA:
                        best_loss = valloss
                        best_OA=valOA
                        Stop_Flag=0
                        torch.save(net.state_dict(),"model\\best_model.pt")
                        torch.save(net,"model\\best_model.pth")
                        print('save model...')
                torch.cuda.empty_cache()
                net.train()
        toc1 = time.perf_counter()
        print("\n\n====================training done. starting evaluation...========================\n")
        
        training_time=toc1 - tic1  
        Train_Time_ALL.append(training_time)
        
        torch.cuda.empty_cache()
        with torch.no_grad():
            net.load_state_dict(torch.load("model\\best_model.pt"))
            net.eval()
            tic2 = time.perf_counter()
            output = net(net_input)
            toc2 = time.perf_counter()
            testloss = compute_loss(output, test_samples_gt_onehot, test_label_mask)
            testOA = evaluate_performance(output, test_samples_gt, test_samples_gt_onehot,require_AA_KPP=True,printFlag=False)
            print("{}\ttest loss={}\t test OA={}".format(str(i + 1), testloss, testOA))
            
            classification_map=torch.argmax(output, 1).reshape([height,width]).cpu()+1 
            mask = np.zeros(gt.shape, dtype='bool')
            ignored_labels = [0]
            for l in ignored_labels:
                mask[gt == l] = True 
            classification_map[mask] = 0 
            Draw_Classification_Map(classification_map, model_dir + dataset_name + str(testOA))
            testing_time=toc2 - tic2  
            Test_Time_ALL.append(testing_time)
           
            
        torch.cuda.empty_cache()
        del net

    OA_ALL = np.array(OA_ALL)
    AA_ALL = np.array(AA_ALL)
    KPP_ALL = np.array(KPP_ALL)
    AVG_ALL = np.array(AVG_ALL)
    Train_Time_ALL=np.array(Train_Time_ALL)
    Test_Time_ALL=np.array(Test_Time_ALL)

    

    print("\ntrain_ratio={}".format(curr_train_ratio),
          "\n==============================================================================")
    print('OA=', np.mean(OA_ALL), '+-', np.std(OA_ALL))
    print('AA=', np.mean(AA_ALL), '+-', np.std(AA_ALL))
    print('Kpp=', np.mean(KPP_ALL), '+-', np.std(KPP_ALL))
    print('AVG=', np.mean(AVG_ALL, 0), '+-', np.std(AVG_ALL, 0))
    print("Average training time:{}".format(np.mean(Train_Time_ALL)))
    print("Average testing time:{}".format(np.mean(Test_Time_ALL)))
    
    
    f = open(model_dir + dataset_name + '_results.txt', 'a+')
    str_results = '\n\n************************************************' \
    +"\ntrain_ratio={}".format(curr_train_ratio) \
    +'\nOA='+ str(np.mean(OA_ALL))+ '+-'+ str(np.std(OA_ALL)) \
    +'\nAA='+ str(np.mean(AA_ALL))+ '+-'+ str(np.std(AA_ALL)) \
    +'\nKpp='+ str(np.mean(KPP_ALL))+ '+-'+ str(np.std(KPP_ALL)) \
    +'\nAVG='+ str(np.mean(AVG_ALL,0))+ '+-'+ str(np.std(AVG_ALL,0)) \
    +"\nAverage training time:{}".format(np.mean(Train_Time_ALL)) \
    +"\nAverage testing time:{}".format(np.mean(Test_Time_ALL)) 
    f.write(str_results)
    f.close()
        
