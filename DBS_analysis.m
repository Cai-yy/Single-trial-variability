% This file is used to analyze the DBS data, plot single neuron tunings,
% show populational activities

clear all;
close all;
addpath('util\');

%% Load Calcium activities and triggers
datafolder='dataset\';
f=10; % The imaging frequency
dT=1/f; 
load(fullfile(datafolder,'DBS_calcium.mat'));
load(fullfile(datafolder,'DBS_trigger.mat'));
load('cortical_out_line_resize_5.mat');
load(fullfile(datafolder,'DBS_motion_energy.mat'));

%% Preprocess the data
% Calcium
valid_Cd=normalize(valid_C,2); % z-scored
valid_Cd=detrend(valid_Cd')';
[nN,~]=size(valid_Cd);
nT=min(size(valid_Cd,2),length(stimuli_array_with_label));
nS=length(start_edge); % Number of stimulation
MIdx=[0,0,1,0,1,0,0,0,1,0,0,1]; % If moving index, 0 for static, 1 for moving, divided by the behavior recording

%% Get the pre-post stimulus activity
tPost=20; % Analyze time window post-stimulus
tPre=10; % Analyze time window pre-stimulus
StiLength=round(mean(end_edge-start_edge));
Pre=zeros(nN,nS); % z-scored
Post=zeros(nN,nS); % z-scored
for inn=1:nN
    for isa=1:nS
        Pre(inn,isa)=mean(valid_Cd(inn,start_edge(isa)-tPre:start_edge(isa)-1)); % Pre response, z-scored
        Post(inn,isa)=mean(valid_Cd(inn,start_edge(isa):start_edge(isa)+tPost)); % Post response, z-scored
    end
end

% Arrange the neurons according to the mean response
mI=mean((Post-Pre),2);
[~,I]=sort(mI,'descend');

%% Visualize the single neuron tuning curve
[~,Ia]=sort(mI,'descend');
[~,Id]=sort(mI,'ascend');

figure();
nP=5; % Number of neurons to plot 
tpre=2;
tpost=5;
% Activated
for inn=1:nP
    axA(inn)=subplot(2,nP,inn);
    for iss=1:min(10,nS)
        cFr=valid_Cd(Ia(inn),start_edge(iss)-tpre/dT:end_edge(iss)+tpost/dT);
        cFr=cFr-mean(cFr(1:tpre/dT));
        plot((-tpre/dT:StiLength+tpost/dT)*dT,cFr,'LineWidth',0.5,'Color',[0.5,0.5,0.5]);
        hold on;
        if iss==1
            mFr=cFr;
        else
            mFr=mFr+cFr;
        end
    end
    mFr=mFr/min(10,nS);
    plot((-tpre/dT:StiLength+tpost/dT)*dT,mFr,'LineWidth',1,'Color',[1,0,0]);
    plot([0,0],ylim(),'--k');
    if inn==1
        xlabel('Time');
        ylabel('dF');
    else
        set(gca,'YTick',[]);
        set(gca,'XTick',[]);
        axis off;
    end
end
linkaxes(axA);

% Deactivated
for inn=1:nP
    axD(inn)=subplot(2,nP,inn+nP);
    for iss=1:min(10,nS)
        cFr=valid_Cd(Id(inn),start_edge(iss)-tpre/dT:end_edge(iss)+tpost/dT);
        cFr=cFr-mean(cFr(1:tpre/dT));
        plot((-tpre/dT:StiLength+tpost/dT)*dT,cFr,'LineWidth',0.5,'Color',[0.5,0.5,0.5]);
        hold on;
        if iss==1
            mFr=cFr;
        else
            mFr=mFr+cFr;
        end
    end
    mFr=mFr/min(10,nS);
    plot((-tpre/dT:StiLength+tpost/dT)*dT,mFr,'LineWidth',1,'Color',[1,0,0]);
    plot([0,0],ylim(),'--k');
    if inn==1
        xlabel('Time');
        ylabel('dF');
    else
        set(gca,'YTick',[]);
        set(gca,'XTick',[]);
        axis off;
    end
end
linkaxes(axD);

%% Populational activity, divide the quit and the locomotive trials
figure();
set(gcf,'position',[100,100,1000,600]);  
names={'Q','L'};
for isa=1:nS
    subplot(1,nS,isa);
    imagesc([start_edge(isa)-StiLength,end_edge(isa)+4*StiLength]/f,[1,nN], ...
        valid_Cd(I,start_edge(isa)-StiLength:end_edge(isa)+4*StiLength));
    hold on;
    yl=ylim;
    plot([start_edge(isa)/f,start_edge(isa)/f],yl,'k--');
%         plot([end_edge(isti),end_edge(isti)],yl,'g--');        
    caxis([0,3]);
    title(['# ',num2str(isa)]);   
    colormap(flipud(othercolor('RdYlBu9')));
    if isa~=1
        set (gca,'ytick',[]);
    end
    title(names{MIdx(isa)+1})
end

%% Plot the motion energy
figure();
for isa=1:nS
    ax(isa)=subplot(1,nS,isa);
    plot((start_edge(isa)-StiLength:end_edge(isa)+4*StiLength)/f,ME(start_edge(isa)-StiLength:end_edge(isa)+4*StiLength),'k');
    hold on;
    ylim([0,20]);
    plot([start_edge(isa)/f,start_edge(isa)/f],[0,20],'--','Color',[0.5,0.5,0.5]);
    axis off;
end
linkaxes(ax,'y');

%% Plot the correlation matrix
% Realign the stimulus according to static 1-8, moving 1-4
Isr=[find(MIdx==0),find(MIdx==1)];% Realigned index of stimuli
Prer = Pre(:,Isr);
Postr = Post(:,Isr);
CM=[Prer,Postr];
rho=corr(CM);
figure();
imagesc(rho);
colormap(othercolor('PuBuGn7'));
axis equal;
caxis([0,0.8]);
colorbar;
hold on;
plot([1,1]*(sum(MIdx==0)+0.5),[0.5,nS*2+0.5],'--k');
plot([0.5,nS*2+0.5],[1,1]*(sum(MIdx==0)+0.5),'--k');
plot([1,1]*(nS+0.5),[0.5,nS*2+0.5],'--k');
plot([0.5,nS*2+0.5],[1,1]*(nS+0.5),'--k');
plot([1,1]*(sum(MIdx==0)+0.5+nS),[0.5,nS*2+0.5],'--k');
plot([0.5,nS*2+0.5],[1,1]*(sum(MIdx==0)+0.5+nS),'--k');
