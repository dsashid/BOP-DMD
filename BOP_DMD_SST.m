%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Simple example --- how to call code
%
% Here we fit data generated from 3 
% spatial modes, each with time dynamics 
% which are exponential in time
%
% The examples show how to call the optdmd
% wrapper with various options
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% load data
clear; clc;
addpath('/Users/nathankutz/Dropbox (uwamath)/boostedDMD/optdmd-master');
load('SST_data.mat');




t0= 0;
t1 = 1000; %ask about this
nt = 1000;
ts = linspace(t0,t1,nt);



%% try bootstrapping - need to decide with or without replacement
%set seed
% rng(0);

%should I normalize? xdata-mean(xdata,1).*ones(89351,1);
%data 
xdata = Z;
m = 360;
n= 180;

% xdata = (xdata-mean(xdata,1).*ones(89351,1))./std(xdata,0,1);
training_cutoff = 1000;

xdata_test = xdata(:,training_cutoff+1:end);
xdata = xdata(:,1:training_cutoff);

%number of time points
[m_space, n_time] = size(xdata);

num_modes = 5;

%size of batch
p = 500;


%number of cycles for each noise cycle
num_cycles =  100;


%create lambda vec for DMD cycles
lambda_vec_DMD = zeros(num_modes,1);

%create lambda vec for optdmd cycles
lambda_vec_optDMD = zeros(num_modes,1);


%create lambda vec for optdmd cycles
lambda_vec_mean_ensembleDMD = zeros(num_modes,1);

%create lambda vec for ensembleDMD cycle
lambda_vec_ensembleDMD = zeros(num_modes,num_cycles);
w_vec_ensembleDMD = zeros(m_space,num_cycles*num_modes);
b_vec_ensembleDMD = zeros(num_modes,num_cycles);



%try DMD
[phi_DMD, lam_DMD, b_DMD, sig_DMD]= DMD(xdata(:,1:end-1), xdata(:,2:end), num_modes);

%should I normalize xdata?

%try regular optdmd
[w_opt,e_opt,b_opt] = optdmd(xdata,ts,num_modes,2, varpro_opts('ifprint',0));


for j = 1:num_cycles
        %try with ioptdmd with DMD modes/evals as IC
        %select indices
        unsorted_ind = randperm(n_time,p);
        %sort ind so in ascending order. NOTE: evals have variable delta t
        ind = sort(unsorted_ind);

        %create dataset for this cycle by taking aforementioned indices
        xdata_cycle = xdata(:,ind);
        %selected index times
        ts_ind = ts(ind);

        [w_cycle,e1_cycle,b_cycle] = optdmd(xdata_cycle,ts_ind,num_modes,2,varpro_opts('ifprint',0),e_opt);

        
%         [b_sort,b_ind]= sort(b_cycle,'descend');

%sort by evals instead
        [sorted,imag_ind]=sort(imag(e1_cycle));
        
        b_vec_ensembleDMD(:,j) = b_cycle(imag_ind);
        lambda_vec_ensembleDMD(:,j) = e1_cycle(imag_ind);
        w_vec_ensembleDMD(:,(j-1)*num_modes+1:j*num_modes) = w_cycle(:,imag_ind);
   
%         b_vec_ensembleDMD(:,j) = b_sort;
%         w_vec_ensembleDMD(:,(j-1)*num_modes+1:j*num_modes) = w_cycle(:,b_ind);
%         lambda_vec_ensembleDMD(:,j) = e1_cycle(b_ind);
        
        %do this!!


       
       
end
%%

%break up lambda and sort?
% [sorted_lambda,imag_ind] = sort(imag(lambda_vec_ensembleDMD),'ascend');
% lambda_vec_ensembleDMD(imag_ind);
% clip= sortedLambda_ensembleDMD(2:3,:);
% clip(sign_ind);

mean_modes = zeros(m_space,num_modes);

%figures for modes and variance
for i =1:num_modes
    w_sample = w_vec_ensembleDMD(:,i:num_modes:end);
    mean_modes(:,i) = mean(w_sample,2);
    
    figure(i); imagesc(imrotate(flipud(reshape(mean(abs(w_sample),2),m,n)),-90));
    caxis([0 0.02])
    c = colorbar;
    c.FontSize = 16;
    c.TickLabelInterpreter  = 'latex';
    xticks([])
    yticks([])
    
    set(gcf,'PaperUnits','inches','PaperPosition',[0 0 6 2.5])

    figure(i+10);imagesc(imrotate(flipud(reshape(abs(var(w_sample,0,2)),m,n)),-90)); 
    c = colorbar;
    c.FontSize = 16;
    c.TickLabelInterpreter  = 'latex';
    xticks([])
    yticks([])
    
    set(gcf,'PaperUnits','inches','PaperPosition',[0 0 6 2.5])
    
end

%% temporal uncertainty

for j = 1:num_modes
    figure(j+20)
    histfit(abs(lambda_vec_ensembleDMD(j,:)),30)
    a = get(gca,'XTickLabel');  
    set(gca,'XTickLabel',a,'fontsize',20)
    a = get(gca,'YTickLabel');  
    set(gca,'YTickLabel',a,'fontsize',20,'TickLabelInterpreter','latex')
end


%% forecasting

pdf_lambda_vec = zeros(2,num_modes);
pdf_b_vec = zeros(2,num_modes);


for jj = 1:num_modes
    %make b_i distributions
    figure(jj);
    hist(b_vec_ensembleDMD(jj,:))
    pd_b = fitdist(b_vec_ensembleDMD(jj,:)','Normal');
    pdf_b_vec(1,jj) = pd_b.mu;
    pdf_b_vec(2,jj) = pd_b. sigma;
   
    %make lambda distributions
    figure(jj+10);
    histfit(abs((lambda_vec_ensembleDMD(jj,:))))
    pd_real = fitdist(real(lambda_vec_ensembleDMD(jj,:)'),'Normal');
    pdf_lambda_vec_real(1,jj) = pd_real.mu;
    pdf_lambda_vec_real(2,jj) = pd_real. sigma;
    
    %store imaginary comps
    pd_imag = fitdist(abs(imag(lambda_vec_ensembleDMD(jj,:)')),'Normal');
    pdf_lambda_vec_imag(1,jj) = pd_imag.mu;
    pdf_lambda_vec_imag(2,jj) = pd_imag. sigma;
end

%use w_opt for now


%test with optdmd
% mean(w_opt*diag(b_opt)*exp(e_opt*(1:1400)),1);

mean_b = mean(b_vec_ensembleDMD,2);




%% test with reconstruction
mean_lambda = mean(lambda_vec_ensembleDMD,2);

%try with mean modes/mean b
x_train =mean_modes*diag(mean_b)*exp(mean_lambda*(1:1000));

plot(1:1000, mean(x_train,1));
hold on;



%%


last_training_point= x_train(:,1000);

%regress to t=100 and set it to 0
new_b = mean_modes*diag(exp(mean_lambda*(1)))\last_training_point;

num_trials = 100;
lambda_sample_real = zeros(num_modes,num_trials);
lambda_sample_imag = zeros(num_modes,num_trials);

b_sample= zeros(num_modes,num_trials);
%now try with BOP-DMD
for r=1:3
    %figure out how to deal with conjugates
    b_sample(r,:) = normrnd(pdf_b_vec(1,r),pdf_b_vec(2,r),[1,num_trials]);

    lambda_sample_real(r,:) = normrnd(pdf_lambda_vec_real(1,r),pdf_lambda_vec_real(2,r),[1,num_trials]);
    lambda_sample_imag(r,:) = normrnd(-1*pdf_lambda_vec_imag(1,r),pdf_lambda_vec_imag(2,r),[1,num_trials])*1i;

    %use w_opt for now
end

lambda_sample = lambda_sample_real+lambda_sample_imag;

lambda_sample(4,:) = conj(lambda_sample(2,:));
lambda_sample(5,:) = conj(lambda_sample(1,:));
% 
b_sample(4,:) = b_sample(2,:);
b_sample(5,:) = b_sample(1,:);


%now forecast
for jj = 1:length(lambda_sample)

forecasted_sample = mean_modes*diag(new_b)*exp(lambda_sample(:,jj)*(1:400));
hold on ;
plot(1001:1400,mean(forecasted_sample,1));
end 

figure(31);
lastpredicted_snapshot = zeros(m_space,length(lambda_sample));
%now forecast
for jj = 1:length(lambda_sample)

forecasted_sample = mean_modes*diag(b_sample(:,jj))*exp(lambda_sample(:,jj)*(1001:1400));
hold on ;
plot(1001:1400,mean(forecasted_sample,1));

lastpredicted_snapshot(:,jj) = forecasted_sample(:,end);
end 


figure(32); imagesc(imrotate(flipud(reshape(abs(mean(lastpredicted_snapshot,2)),m,n)),-90));
c = colorbar;
c.FontSize = 16;
c.TickLabelInterpreter  = 'latex';
xticks([])
yticks([])
caxis([0 30])
colorbar off

figure(33); imagesc(imrotate(flipud(20*reshape(std(abs(lastpredicted_snapshot),0,2),m,n)),-90));
c = colorbar;
c.FontSize = 16;
c.TickLabelInterpreter  = 'latex';
xticks([])
yticks([])
caxis([0 30])
%colorbar off

%%
% plot actual snapshot
figure(35); imagesc(imrotate(flipud(reshape(abs(xdata(:,end)),m,n)),-90));
c = colorbar;
c.FontSize = 16;
c.TickLabelInterpreter  = 'latex';
xticks([])
yticks([])
caxis([0 30])
colorbar off


%try doing exact DMD last snapshot
deltat = 1;
Lambda_continuous = diag(log(diag(lam_DMD))/deltat);

for time = 1:1400
x_future_continuous(:,time) = phi_DMD* expm((Lambda_continuous)*time)*b_DMD;
end

plot(1:1400, mean(x_future_continuous,1));


figure(43); imagesc(imrotate(flipud(reshape(abs(x_future_continuous(:,end)),m,n)),-90));
c = colorbar;
caxis([0 30])
c.FontSize = 16;
c.TickLabelInterpreter  = 'latex';
xticks([])
yticks([])


%try doing exact DMD last snapshot
deltat = 1;
Lambda_continuous = diag(log(diag(lam_DMD))/deltat);

for time = 1:1400
x_future_continuous(:,time) = phi_DMD* expm((Lambda_continuous)*time)*b_DMD;
end

figure(44);
plot(1:1400, mean(x_future_continuous,1));


figure(43); imagesc(imrotate(flipud(reshape(abs(x_future_continuous(:,end)),m,n)),-90));
c = colorbar;
caxis([0 30])
c.FontSize = 16;
c.TickLabelInterpreter  = 'latex';
xticks([])
yticks([])




