clc
clear all
close all
%% Load data and rescale to [-1 1]
load('laser_dataset.mat')
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
%% Model Selection
Nu=1;
Ny=1;
Nh=[50, 100, 500, 1000]; 
omega_in=[1, 3, 5];
rho=[0.9, 0.95, 1];
lambda=[0, 0.1, 0.01];
Nw=[500, 1000, 2000];
e_thr=inf;
iteration=0;

for Nhi=1:length(Nh)
    for omegai=1:length(omega_in)
        for rhoi=1:length(rho)
            for lambdai=1:length(lambda)
              for Nwi=1:length(Nw)  
               
                U = 2*rand(Nh(Nhi),Nu)-1;    
                U = omega_in (omegai) * U; 
                W = 2*rand(Nh(Nhi),Nh(Nhi)) - 1;
                W = rho (rhoi) * (W / max(abs(eig(W))));
                state = zeros(Nh(Nhi),1);
                H=zeros(Nh(Nhi),length(x_train));
                
                    for t = 1:length(x_train)
                    state = tanh(U * x_train(t) + W * state);
                    H(:,t) = state;
                    end
                
                H=H(:,Nw(Nwi)+1:end);
                D=y_train(:,Nw(Nwi)+1:end);
                V = D*H'*inv(H*H'+ lambda(lambdai) *eye(Nh(Nhi)));
               
                Y_tr = V * H;
                E_tr = immse(D,Y_tr);
                
                for t = 1:length(x_validation)
                    state = tanh(U * x_validation(t) + W * state);
                    Y_vl(:,t) = V * state;
                end

                E_vl(iteration)= immse(y_validation,Y_vl)
                
                if E_vl(iteration)< e_thr
                   Nh_best= Nh(Nhi);
                   omega_in_best=omega_in(omegai);
                   rho_best=rho(rhoi);
                   lambda_best=lambda(lambdai);
                   Nw_best=Nw(Nwi);
                end
                e_thr=E_vl(iteration);
            end
        end
    end
    end
end

hyperparameters.Nh_best=Nh_best;
hyperparameters.omega_in_best=omega_in_best;
hyperparameters.rho_best=rho_best;
hyperparameters.lambda_best=lambda_best;
hyperparameters.Nw_best=Nw_best;
save('hyperparameters.mat','hyperparameters')