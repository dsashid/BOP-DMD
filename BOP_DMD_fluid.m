%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%Code for BOP DMD on vortex shedding example.
%Requirements: ALL.mat and optDMD packages
%Author: Diya Sashidhar
%Date modified: Jun 29, 2021
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% load data

load('ALL.mat');


%% 

%data 
xdata = VORTALL;

% %add noise - run noisetofluids.m
% noise_matrix = added_noise.*ones(89351,151);
% xdata = xdata+noise_matrix ;

%EDIT: 6/29 removed standardization
% standardize
% xdata= (xdata-mean(xdata,1).*ones(89351,1))./std(xdata,0,1);

training_cutoff = 100;

xdata_test = xdata(:,training_cutoff+1:end);
xdata = xdata(:,1:training_cutoff);


%number of time points
[m_space, n_time] = size(xdata);

t0= 0;
t1 = n_time; 
nt = n_time; %ask nathan what unit of time is
ts = linspace(t0,t1,nt);


num_modes = 9;

%size of batch
p = 30;


%number of cycles 
num_cycles =  100;


%create lambda matrix: vector of eigenvalues for each ensembleDMD cycle
lambda_vec_ensembleDMD = zeros(num_modes,num_cycles);

%create eigenvector matrix: concatenated matrix of eigenvectors  for each ensembleDMD cycle
w_vec_ensembleDMD = zeros(m_space,num_cycles*num_modes);

%create b matrix: amplitude vector for each ensemble DMD cycle
b_vec_ensembleDMD = zeros(num_modes,num_cycles);




%try regular optdmd: for initial seed
[w_opt,e_opt,b_opt] = optdmd(xdata,ts,num_modes,2, varpro_opts('ifprint',0));


for j = 1:num_cycles
        %try with optdmd with optDMD modes/evals as IC
        %randomly select p indices from n time points
        unsorted_ind = randperm(n_time,p);
        
        %sort ind so in ascending order. NOTE: evals have variable delta t
        ind = sort(unsorted_ind);

        %create dataset for this cycle by taking aforementioned indices
        xdata_cycle = xdata(:,ind);
        
        %selected index times
        ts_ind = ts(ind);
        
        %run with optDMD eigenvalues as IC and generate evals/evecs for
        %each cycle
        [w_cycle,e1_cycle,b_cycle] = optdmd(xdata_cycle,ts_ind,num_modes,2,varpro_opts('ifprint',0),e_opt);

        

     %sort by imaginary component of evals so that it's easier to sort
        [sorted,imag_ind]=sort(imag(e1_cycle));
        
        b_vec_ensembleDMD(:,j) = b_cycle(imag_ind);
        lambda_vec_ensembleDMD(:,j) = e1_cycle(imag_ind);
        w_vec_ensembleDMD(:,(j-1)*num_modes+1:j*num_modes) = w_cycle(:,imag_ind);
   
end

%% generate figs

%initial mean mode matrix 
mean_modes = zeros(m_space,num_modes);

for i =1:num_modes
    
    %take eigenvectors for each mode
    w_sample = w_vec_ensembleDMD(:,i:num_modes:end);
    %calculate mean of each mode
    mean_modes(:,i) = mean(w_sample,2);

    %plot mean absolue value of each mode
    figure(i);pcolor(reshape(mean(abs(w_sample),2),m,n)); shading interp;
    c = colorbar;
    c.FontSize = 16;
    c.TickLabelInterpreter  = 'latex';
    xticks([])
    yticks([])

    %plot var of each mode 
%     figure(i+10);pcolor(reshape(abs(var(w_sample,0,2)),m,n));shading interp; 
%     c = colorbar;
%     c.FontSize = 16;
%     c.TickLabelInterpreter  = 'latex';
%     xticks([])
%     yticks([])
%     
end



%% temporal uncertainty: plot histograms of eigenvals for each mode

for j = 1:num_modes
    figure(j+20)
    histfit(abs(lambda_vec_ensembleDMD(j,:)),30)
    a = get(gca,'XTickLabel');  
    set(gca,'XTickLabel',a,'fontsize',20)
    a = get(gca,'YTickLabel');  
    set(gca,'YTickLabel',a,'fontsize',20,'TickLabelInterpreter','latex')
end


%% generate distributions for forecasting


for jj = 1:num_modes
    %make b_i distributions
%     figure(jj);
%     hist(b_vec_ensembleDMD(jj,:))
    pd_b = fitdist(b_vec_ensembleDMD(jj,:)','Normal');
    pdf_b_vec(1,jj) = pd_b.mu;
    pdf_b_vec(2,jj) = pd_b. sigma;
   
    %make lambda distributions for real component
    figure(jj+10);
    histfit(abs((lambda_vec_ensembleDMD(jj,:))),20)
    pd_real = fitdist(real(lambda_vec_ensembleDMD(jj,:)'),'Normal');
    pdf_lambda_vec_real(1,jj) = pd_real.mu;
    pdf_lambda_vec_real(2,jj) = pd_real. sigma;
    
    %store imaginary comps for imaginary component
    pd_imag = fitdist(abs(imag(lambda_vec_ensembleDMD(jj,:)')),'Normal');
    pdf_lambda_vec_imag(1,jj) = pd_imag.mu;
    pdf_lambda_vec_imag(2,jj) = pd_imag. sigma;
end



%%  reconstruction

%calculate mean value for all lambda's generated from ensemble DMD

mean_lambda = mean(lambda_vec_ensembleDMD,2);


%calculate mean value for all b's generated from ensemble DMD
mean_b = mean(b_vec_ensembleDMD,2);


%reconstruct: try with mean modes/mean b for training data
x_train =mean_modes*diag(mean_b)*exp(mean_lambda*(0:99));



%%

last_training_point= x_train(:,100);

%regress to t=100 and set it to t =0 to get a new b
new_b = mean_modes*diag(exp(mean_lambda*(1)))\last_training_point;

%number of trials for forecasting
num_trials = 100;

%initialize real/imag eigenvalues and b for drawing 
lambda_sample_real = zeros(num_modes,num_trials);
lambda_sample_imag = zeros(num_modes,num_trials);
b_sample= zeros(num_modes,num_trials);



%hacky way to deal with conjugates
for r=1:5
    b_sample(r,:) = normrnd(pdf_b_vec(1,r),pdf_b_vec(2,r),[1,num_trials]);

    lambda_sample_real(r,:) = normrnd(pdf_lambda_vec_real(1,r),pdf_lambda_vec_real(2,r),[1,num_trials]);
    lambda_sample_imag(r,:) = normrnd(-1*pdf_lambda_vec_imag(1,r),pdf_lambda_vec_imag(2,r),[1,num_trials])*1i;

    %use w_opt for now
end

%combine to create a complex eigenvalue with conjugate pairs
lambda_sample = lambda_sample_real+lambda_sample_imag;

lambda_sample(6,:) = conj(lambda_sample(4,:));
lambda_sample(7,:) = conj(lambda_sample(3,:));
lambda_sample(8,:) = conj(lambda_sample(2,:));
lambda_sample(9,:) = conj(lambda_sample(1,:));


b_sample(6,:) = b_sample(4,:);
b_sample(7,:) = b_sample(3,:);
b_sample(8,:) = b_sample(2,:);
b_sample(9,:) = b_sample(1,:);



forecasted_sample = zeros(num_trials,53);
%now forecast
for jj = 1:length(lambda_sample)

forecasted_sample(jj,:) = mean(mean_modes*diag(new_b)*exp(lambda_sample(:,jj)*(0:52)),1);

end 


figure(40)
plot(1:100, mean(x_train,1),'Linewidth',2);
hold on;

plot(1:100,mean(xdata,1),'Linewidth',2);

plot(99:151,forecasted_sample,'k', 'Linewidth',2);
hold on;
plot(99:151,max(real(forecasted_sample), [],1),'r','Linewidth',2);
plot(99:151,min(real(forecasted_sample), [],1),'b','Linewidth',2);

%%

figure(31);
lastpredicted_snapshot = zeros(m_space,length(lambda_sample));

%now forecast for the next 50 time points (without regressing)
for jj = 1:length(lambda_sample)

forecasted_sample = mean_modes*diag(b_sample(:,jj))*exp(lambda_sample(:,jj)*(101:151));
hold on ;
plot(101:151,mean(forecasted_sample,1));

lastpredicted_snapshot(:,jj) = forecasted_sample(:,end);
end 

plot(101:151,mean(xdata(:,101:151),1),'Linewidth',2);


%% plot mean, std, var of last predicted snapshot as well as actual last snapshot

%plot last predicted snapshot mean
figure(32); imagesc(imrotate(flipud(reshape(abs(mean(lastpredicted_snapshot,2)),m,n)),-90));
c = colorbar;
c.FontSize = 16;
c.TickLabelInterpreter  = 'latex';
xticks([])
yticks([])

%plot last predicted snapshot std
figure(33); imagesc(imrotate(flipud(reshape(std(abs(lastpredicted_snapshot),0,2),m,n)),-90));
c = colorbar;
c.FontSize = 16;
c.TickLabelInterpreter  = 'latex';
xticks([])
yticks([])

%plot last predicted snapshot var
figure(34); imagesc(imrotate(flipud(reshape(var(abs(lastpredicted_snapshot),0,2),m,n)),-90));
c = colorbar;
c.FontSize = 16;
c.TickLabelInterpreter  = 'latex';
xticks([])
yticks([])

% plot actual snapshot
figure(35); imagesc(imrotate(flipud(reshape(abs(xdata(:,end)),m,n)),-90));
c = colorbar;
c.FontSize = 16;
c.TickLabelInterpreter  = 'latex';
xticks([])
yticks([])