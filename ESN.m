clc
clear all
close all
%% Load data and rescale to [-1 1]
load('laser_dataset.mat')
load('hyperparameters.mat')
data=laserTargets;
data_mat=cell2mat(data);
data_mat_normalized=rescale(data_mat,-1,1);
%% properly separate input and target data 
input=data_mat_normalized(1:end-1);
target=data_mat_normalized(2:end);
%% Training, Validation, and Test Split
x_train=input(1:4000);
y_train=target(1:4000);

x_validation=input(4001:5000);
y_validation=target(4001:5000);

x_test=input(5001:end);
y_test=target(5001:end);
%% Operation phase on the selected model
Nh_best=hyperparameters.Nh_best;
omega_in_best=hyperparameters.omega_in_best;
rho_best=hyperparameters.rho_best;
lambda_best=hyperparameters.lambda_best;
Nw_best=hyperparameters.Nw_best;

in_tr=input(1:5000);
out_tr=target(1:5000);
Nu=1;
Ny=1;
epoch=1;
for epoch=1:30
    %%Random intialize, based on best hyperparameters
    U = 2*rand(Nh_best,Nu)-1;    
    U = omega_in_best * U; %input-to-reservoir weight matrix Win(U)
    W = 2*rand(Nh_best,Nh_best) - 1;
    W = rho_best * (W / max(abs(eig(W)))); %recurrent reservoir weight matrix Wr(W)
    state = zeros(Nh_best,1);
    H=zeros(Nh_best,length(in_tr));
    %%Run the reservoir on the input stream                
    for t = 1:length(in_tr)        
             state = tanh(U * in_tr(t) + W * state);
             H(:,t) = state;
    end
    %%Discard the washout                
    H=H(:,Nw_best+1:end);
    D=out_tr(:,Nw_best+1:end);
    %%Train the readout
    V = D*H'*inv(H*H'+ lambda_best *eye(Nh_best));   %reservoir-to-readout weight matrix Wout (V)                
    estimated_out_tr = V * H;
    E_tr_whole(epoch) = immse(D,estimated_out_tr);
    %%Operation Phase, computing the error per epoch for X_test                
    for t = 1:length(x_test)
               state = tanh(U * x_test(t) + W * state);
               estimated_out_ts(:,t) = V * state;
    end
    E_ts_whole(epoch)= immse(y_test,estimated_out_ts);
  
    %%Operation Phase, computing the error per epoch for X_validation
    for t = 1:length(x_validation)
               state = tanh(U * x_validation(t) + W * state);
               estimated_out_vl(:,t) = V * state;
    end    
    E_vl_whole(epoch)= immse(y_validation,estimated_out_vl);
    %ESN Structure
    field1 = 'ESN_no';        
    value1 = epoch; 
    field2 = 'Win';           
    value2 = U;
    field3 = 'Wr';            
    value3 = W;
    field4 = 'Wout';          
    value4 = V;
    ESN_structures(epoch) = struct(field1,value1,field2,value2,field3,value3,field4,value4);
end
% save('ESN_structures.mat','ESN_structures')
%% Average performance on reservoir guesses (number of epoch)
ESN_trMSE=mean(E_tr_whole);
ESN_tsMSE=mean(E_ts_whole);
ESN_vlMSE=mean(E_vl_whole);

save('ESN_trMSE.mat','ESN_trMSE')
% save('ESN_tsMSE.mat','ESN_tsMSE')
% save('ESN_vlMSE.mat','ESN_vlMSE')
%% figures
%%on test
figure; clf;
plot(5001:10092,estimated_out_ts,'r--');
hold on;
plot(5001:10092,y_test,'b--');
legend('Estimated Output','Target')
title('Comparative Plot on Test Set')
set(gca, 'xlim', [5001 10092]);
% saveas(gcf,'test_target_output.png')
% saveas(gcf,'test_target_output')
%%on train
figure; clf;
plot(1:length(estimated_out_tr),estimated_out_tr,'r--');
hold on;
plot(1:length(D),D,'b--');
legend('Estimated Output','Target')
title('Comparative Plot on Training Set')
set(gca, 'xlim', [1 4500]);
% saveas(gcf,'train_target_output.png')
% saveas(gcf,'train_target_output')