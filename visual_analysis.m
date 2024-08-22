% This file is used to analyze the visual stimulation data. Do decoding
% with selected neurons and increasing FOV

clear all;
close all;
addpath('util\');

%% Load everything
datafolder='dataset\';
f=10; % The imaging frequency
dT=1/f; 
load(fullfile(datafolder,'visual_calcium.mat'));
load(fullfile(datafolder,'visual_trigger.mat'));
load('cortical_out_line_resize_5.mat');

%% Preprocess
[nN,nT]=size(valid_C);
valid_Cd=detrend(valid_C')';   
valid_Cd=normalize(valid_Cd,2);
nS=length(start_edge);

%% Decode with increased FOV
% Define the decoding parameters
opt.Dcwd=20; % The decoding time window
nB=opt.Dcwd;

% Define the center of visual neurons based on the actual neural number
Mx=228; % The center of brain in x axis
IL=find(valid_neuron_x>Mx);
va=generateAreaid(valid_neuron_x,valid_neuron_y);
Ta=tabulate(categorical(va));
anames=id2acro(str2double(Ta(:,1)));
kvis=[];
for iaa=1:length(anames)
    if startsWith(anames{iaa},'VISp') 
        kvis(end+1)=iaa;
    end
end
inVisL=ismember(va,str2double(Ta(kvis,1))) & (valid_neuron_x>Mx); % If in the left visual area
cVisx=mean(valid_neuron_x(inVisL)); 
cVisy=mean(valid_neuron_y(inVisL));
cVis=[cVisx,cVisy];
% Visualize the calculated center of visual cortex
Plot_neuron_posi_measure_thres([valid_neuron_x;cVisx],[valid_neuron_y;cVisy],[zeros(1,nN),1],10,jet,0.5,0.5,1);

VcenterL=cVis;

D2V=sqrt(sum(([valid_neuron_x,valid_neuron_y]-VcenterL).^2,2)); % Distance to left visual cortex
D2V(~IL)=Inf; % Use left hemisphere 
[~,Id2v]=sort(D2V,'ascend'); 

nNL=sum(valid_neuron_x>Mx);
nL=[1,2,3,4,5,6,7,8,9,10,15,20,30:10:200,250:50:1000,...
    1500:200:nNL];
nL(nL>nNL)=[];
nL=unique(nL);
corrLo=zeros(size(nL));
mseLm=zeros(size(nL));
fovdL=zeros(size(nL));

parfor kk=1:length(nL) 
    nUD = nL(kk);
    Iin=Id2v(1:nUD);
    fovd=D2V(Iin(end));

    % Decode the visual orientation
    X=zeros(nS,nUD*nB);
    y=zeros(nS,1);
    for iss=1:nS
        R=valid_Cd(Iin,start_edge(iss):start_edge(iss)+opt.Dcwd-1); 
%         R=R-mean(valid_Cd(Iin,start_edge(iss)-10:start_edge(iss)),2);
        R=imresize(R,[nUD,nB]);
        R=reshape(R,1,[]);
        X(iss,:)=R;
        y(iss)=stimuli_array_with_label(start_edge(iss)+1);
    end
    cvMCR=crossval('mcr',X,y,'Predfun',@mknn,'leaveout',1);
    corrLo(kk)=1-cvMCR;
    fovdL(kk)=fovd;
    fprintf('fov= %f, number of neuron = %d, correctness= %f \n',fovd,nUD,1-cvMCR);    
end

figure();
subplot(2,1,1);
plot(nL,corrLo,'LineWidth',1.5,'Color','black');
xlabel('Number of neurons');
ylabel('Orientation decoding accuracy');
xlim([0,nNL]);
set(gca,'XScale','log');
subplot(2,1,2);
plot(fovdL*(4.4/(247-74)),corrLo,'LineWidth',1.5,'Color','blue');
xlabel('FOV radius (mm)');
ylabel('Orientation decoding accuracy');
set(gca,'XScale','log');

%% Decode with selected neurons
Nlist=[1:300,320:20:500,600:100:5000,5500:500:3*1e4];
Nlist(Nlist>nN)=[];
% Use few data to calculate the variablity
pSlct=0.3; % percentage of trials used to calculate variability
nSSlct=round(nS*pSlct);
nSDcd=nS-nSSlct;
TDcd=round(nT*(1-pSlct));


% See the decoding accuracy of each neuron
% Decode the visual orientation
accNL=zeros(nN,1);
parfor inn=1:nN
    X=zeros(nSSlct,nB);
    y=zeros(nSSlct,1);
    for iss=1:nSSlct
        R=valid_Cd(inn,start_edge(iss):start_edge(iss)+opt.Dcwd-1); 
        R=imresize(R,[1,nB]);
        R=reshape(R,1,[]);
        X(iss,:)=R;
        y(iss)=stimuli_array_with_label(start_edge(iss)+1);
    end
    cvMCR=crossval('mcr',X,y,'Predfun',@mknn,'leaveout',1);
    accNL(inn)=1-cvMCR;
    
    if mod(inn,100)==0
        fprintf('%d / %d \n',inn,nN);
    end

end

[~,Ia]=sort(accNL,'descend');

% Decode with increased selected data
% Decode the visual orientation
accLp=zeros(size(Nlist)); % Accuracy list using picked neurons
parfor kk=1:length(Nlist)
    nUD=Nlist(kk);
    X=zeros(nSDcd,nUD*nB);
    y=zeros(nSDcd,1);
    for iss=nSSlct+1:nS
        R=valid_Cd(Ia(1:nUD),start_edge(iss):start_edge(iss)+opt.Dcwd-1); 
        R=imresize(R,[nUD,nB]);
        R=reshape(R,1,[]);
        X(iss-nSSlct,:)=R;
        y(iss-nSSlct)=stimuli_array_with_label(start_edge(iss)+1);
    end
    cvMCR=crossval('mcr',X,y,'Predfun',@mknn,'leaveout',1);
    accLp(kk)=1-cvMCR;
    fprintf('number of neuron = %d, correctness= %f \n',nUD,1-cvMCR); 
end

% Decode with randomly picked neurons
nRd=5; % Number of random times
accLr=zeros(nRd,length(Nlist)); % Accuracy list of decode using randomly selected neurons
for iir=1:nRd
    tmp=1:nN;
    tmp=tmp(randperm(nN));
    parfor kk=1:length(Nlist)
        nUD=Nlist(kk);         
        Iuse=tmp(1:nUD);
        X=zeros(nSDcd,nUD*nB);
        y=zeros(nSDcd,1);
        for iss=nSSlct+1:nS
            R=valid_Cd(Iuse,start_edge(iss):start_edge(iss)+opt.Dcwd-1); 
            R=imresize(R,[nUD,nB]);
            R=reshape(R,1,[]);
            X(iss-nSSlct,:)=R;
            y(iss-nSSlct)=stimuli_array_with_label(start_edge(iss)+1);
        end
        cvMCR=crossval('mcr',X,y,'Predfun',@mknn,'leaveout',1);
        accLr(iir,kk)=1-cvMCR;
        fprintf('number of neuron = %d, correctness= %f \n',nUD,1-cvMCR); 
    end
end


figure();
plot(Nlist,accLp);
hold on;
shadedErrorBar(Nlist,accLr,{@mean @(x) std(x)/nRd});
set(gca,'XScale','log');
legend({'Selected','Random'});
xlabel('Number of neurons');
ylabel('Decoding accuracy');

%% Functions
function yfit=mknn(Xtrain,ytrain,Xtest)
    % Train the predictor
    Mdl = fitcknn(Xtrain,ytrain);
    % Test
    yfit=predict(Mdl,Xtest);
end