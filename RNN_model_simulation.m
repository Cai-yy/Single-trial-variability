% This file is used to generate the dynamics of a recurrent connected
% model with external input. Also, visualize the tuned populational and
% single neuron activities, plot the MDS space and energy landscape

clear all;
close all;
addpath('util\');

rng(2024);

%% Parameters
opt.nNeuron=1000; % number of neuron
opt.g=1.2; % The connection strength
opt.dT=0.1; % The simulation timestep, in sec

opt.TAll=2000; % Total simulation time, in sec
opt.tau=0.5; % Decay costant of RNN units
opt.tauWN=0.2; % Decay costant of noise
opt.StiDur=1; % Stimulus duration, in sec
opt.StiGap=130; % Stimulation gap, in sec
opt.JitterSig=10; % The gap jitter range
opt.StiAmp=1; % StiAmplitude, if a list, the amplitude will be organized in a pseudo-random manner
opt.StiTar=1:150; % Stimulus target neuron
opt.ampInWN=0.09;
opt.SimTimes=1; % The simulation repeat times
opt.Weightdir=[]; % The dir of pre-defined weight matrix, (if not empty, otherwise will be randomly generated)

nN=opt.nNeuron;
g=opt.g;
dT=opt.dT;
outname='simulation_result';
outpath=['output/',outname,'/'];
mkdir(outpath);

%% Simulation parameters
TAll=opt.TAll;
nT=TAll/dT; % The simulation total steps
tau=opt.tau;
StiAmp=opt.StiAmp;
StiTar=opt.StiTar;
nSTy=length(StiAmp); % The number of types of stimulation 

%% External sti
IpS=abs(normrnd(0,0.2,[length(opt.StiTar),1]));
[IpS,~]=sort(IpS,'descend'); % The projection weight
stiSqc=zeros(1,nT);
nS=floor(TAll/(opt.StiDur+opt.StiGap))-1;
nSE=floor(nS/nSTy);
StiArray=repmat(StiAmp,nSE);
StiArray=StiArray(randperm(length(StiArray)));
nS=nSE*nSTy;
sedge=zeros(1,nS);
eedge=zeros(1,nS);
for iss=1:nS
    if iss==1
        estt=opt.StiGap/dT;
    else
        estt=round(eend+(opt.StiDur+rand()*opt.JitterSig-opt.JitterSig/2)/dT+opt.StiGap/dT);
    end
    eend=estt+(opt.StiDur)/dT;
    if eend>nT-100 % If too many, stop
        nS = iss-1;
        StiArray=StiArray(1:nS);
        sedge=sedge(1:nS);
        eedge=eedge(1:nS);
        break;        
    end
    sedge(iss)=estt;
    eedge(iss)=eend;
    stiSqc(estt:eend)=StiArray(iss); % Set a WPM wave
end

E=zeros(nN,nT); 
E(opt.StiTar,:)=IpS*stiSqc;

figure();
plot(stiSqc); title('Stimulation sequence');

%% Run for multiple times
SimTimes=opt.SimTimes;
FrAll=zeros(SimTimes,nN,nT);
SAll=zeros(SimTimes,nN,nT);
SRAll=zeros(SimTimes,nN,nT);
for sstt=1:SimTimes
    % Randomize a connectivity matrix
    J = g * randn(nN,nN) / sqrt(nN); % Connection strength
%     save([outpath,'J',num2str(sstt),'.mat'],'J');

    % set up white noise inputs
    tauWN=opt.tauWN;
    ampInWN=opt.ampInWN;
    ampWN = sqrt((tauWN/dT));
    iWN = ampWN*randn(nN, nT);
    inputWN = zeros(nN,nT);
    for tt = 2: nT
            inputWN(:, tt) = iWN(:, tt) + (inputWN(:, tt - 1) - iWN(:, tt))*exp(-(dT/tauWN));
    end
    inputWN = ampInWN*inputWN; % input noise

    ini=rand(nN,1)*2-1; % The initial membrane potential
    
    % Run the network
    [Fr,~,~]=runSNN_v2(ini,J,nT,tau,E,inputWN,dT);    
    
    fprintf('%d / %d \n',sstt,SimTimes);
end

figure();
imagesc(normalize(Fr,2));

%% Collect the result
% Sort the responsive neurons, as in real data
Pre=zeros(nN,nS); % z-scored
Post=zeros(nN,nS); % z-scored
tPre=10; % Pre stimuli time, in frames
tPost=30; % Post stimuli time, in frames
for inn=1:nN
    for isa=1:nS
        Pre(inn,isa)=mean(Fr(inn,sedge(isa)-tPre:sedge(isa)-1)); % Pre response, z-scored
        Post(inn,isa)=mean(Fr(inn,sedge(isa):sedge(isa)+tPost)); % Post response, z-scored   
    end
end
mI=mean((Post-Pre),2);
[~,I]=sort(mI,'descend');

% Population response, show the first 10 trials
figure();
set(gcf,'position',[100,100,1000,600]);  
% For visualization, normalize Fr
Frn=normalize(Fr,2);
for iss=1:min(10,nS)
    subplot(1,10,iss);
    imagesc([sedge(iss)-20,eedge(iss)+50]*dT,[1,nN],Frn(I,sedge(iss)-20:eedge(iss)+50));
    hold on;
    yl=ylim;
    plot([sedge(iss)*dT,sedge(iss)*dT],yl,'k--');
    caxis([-0.5,2.5]);
    title(['# ',num2str(iss)]);   
    colormap(flipud(othercolor('RdYlBu9')));
    if iss~=1
        set(gca,'YTick',[]);
    end
end

%% Plot the single neuron tuning curve
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
        cFr=Fr(Ia(inn),sedge(iss)-tpre/dT:eedge(iss)+tpost/dT);
        cFr=cFr-mean(cFr(1:tpre/dT));
        plot((-tpre/dT:(opt.StiDur)/dT+tpost/dT)*dT,cFr,'LineWidth',0.5,'Color',[0.5,0.5,0.5]);
        hold on;
        if iss==1
            mFr=cFr;
        else
            mFr=mFr+cFr;
        end
    end
    mFr=mFr/min(10,nS);
    plot((-tpre/dT:(opt.StiDur)/dT+tpost/dT)*dT,mFr,'LineWidth',1,'Color',[1,0,0]);
    if inn==1
        xlabel('Time');
        ylabel('dF');
    else
        set(gca,'YTick',[]);
        set(gca,'XTick',[]);
        axis off;
    end
    if inn==1
        yl=ylim();
    end
    plot([0,0],yl,'--k');
end

% Deactivated
for inn=1:nP
    axD(inn)=subplot(2,nP,inn+nP);
    for iss=1:min(10,nS)
        cFr=Fr(Id(inn),sedge(iss)-tpre/dT:eedge(iss)+tpost/dT);
        cFr=cFr-mean(cFr(1:tpre/dT));
        plot((-tpre/dT:(opt.StiDur)/dT+tpost/dT)*dT,cFr,'LineWidth',0.5,'Color',[0.5,0.5,0.5]);
        hold on;
        if iss==1
            mFr=cFr;
        else
            mFr=mFr+cFr;
        end
    end
    mFr=mFr/min(10,nS);
    plot((-tpre/dT:(opt.StiDur)/dT+tpost/dT)*dT,mFr,'LineWidth',1,'Color',[1,0,0]);
    if inn==1
        xlabel('Time');
        ylabel('dF');
    else
        set(gca,'YTick',[]);
        set(gca,'XTick',[]);
        axis off;
    end
    hold on;

    yl=ylim();

    plot([0,0],yl,'--k');
end
savefig([outpath,'Single neuron tuning curve.fig']);

%% Do MDS dimension reduction
% To save the computational load, use only part of the activty
ituse=[];
tpre=50;
tpost=50;
for isti=1:nS
    ituse=[ituse,sedge(isti)-tpre:eedge(isti)+tpost];
end
dissimilarities=pdist(Fr(:,ituse)','cosine');
[ymds,stress,disparities]=mdscale(dissimilarities,2);
distances=pdist(ymds);
[dum,ord] = sortrows([disparities(:) dissimilarities(:)]);
% plot(dissimilarities,distances,'bo', ...
% dissimilarities(ord),disparities(ord),'r.-');

%% Plot the energy landscape
% For each point in the neural space, calculate its velocity
V=zeros(1,length(ituse)-1); % The velocity vector
ka=1;
figure();
[xq,yq]=meshgrid(linspace(min(ymds(:,1)),max(ymds(:,1)),100), linspace(min(ymds(:,2)),max(ymds(:,2)),100));
e=E(:,sedge(1)+2); % The external input
ns=zeros(nN,1); % Set noise to zeros
kkk=1;
for tt=ituse(1:end-1)
    [~,v]=runSNN_step(Fr(:,tt),J,tau,e,ns,dT);
    V(kkk)=v;
    kkk=kkk+1;
end
tf=isoutlier(V,'percentiles',[0,98]);
xx=ymds(1:end-1,1);
yy=ymds(1:end-1,2);
xx=xx(~tf);
yy=yy(~tf);
V=V(~tf);
F = scatteredInterpolant(xx,yy,V');
F.ExtrapolationMethod = 'none';
P= F(xq,yq);
P = imgaussfilt(P,3);
mesh(xq,yq,P,'FaceColor','flat','EdgeColor','none','FaceAlpha',0.7);
colormap('turbo');
hold on;
% Plot some actual traces onto the map
nStiCur=length(sedge);
for ii=1:nS
    for tt=sedge(ii):eedge(ii)-1
        % Attach the point to the grid
        trans=find(ituse==tt);
        dd1=(xq-ymds(trans,1)).^2+(yq-ymds(trans,2)).^2;
        dd1(isnan(P))=inf; % Do not consider outliers
        [idx11,idx12]=find(dd1==min(dd1,[],'all'));
        dd2=(xq-ymds(trans+1,1)).^2+(yq-ymds(trans+1,2)).^2;
        dd2(isnan(P))=inf; % Do not consider outliers
        [idx21,idx22]=find(dd2==min(dd2,[],'all'));
        plot3([xq(idx11,idx12),xq(idx21,idx22)],[yq(idx11,idx12),yq(idx21,idx22)],[P(idx11,idx12),P(idx21,idx22)],':','LineWidth',1.5,'Color',[0.3,0.3,0.3]);
%         scatter3(xq(idx11,idx12),yq(idx11,idx12),P(idx11,idx12),10,')
        hold on;
    end
end
grid off;
ka=ka+1;
xlabel('MDS axis 1');
ylabel('MDS axis 2');
zlabel('Energy');
    
%% Calculate the time-dependant trial variance
figure();
tPre=10; % In frames
tPost=40; % In frames
stdM=zeros(opt.SimTimes,tPre+tPost);
for ism=1:opt.SimTimes    
    Resp=zeros(nN,tPre+tPost);
    for iss=1:nS
        Resp(:,:,iss)=Fr(:,sedge(iss)-tPre:sedge(iss)+tPost-1);
    end
    stdM(ism,:)=calTdStd(Resp);
end
plot((-tPre:tPost-1)*dT,stdM); hold on;
plot([0,0],ylim(),'--k');
savefig([outpath,'t-trial variance.fig']);
xlabel('Time');
ylabel('Trial variance');

