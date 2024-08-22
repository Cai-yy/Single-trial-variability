% This file is used to calculate the time dependent std

function std=calTdStd(R)

    % R: the response matrix, neuron number x response time x ntrial
    [~,~,nTrial]=size(R);
    meanSR=squeeze(mean(R,3));
    Dis=zeros(size(meanSR));
    for sstt=1:nTrial
        Dis=Dis+(squeeze(R(:,:,sstt))-meanSR).^2;
    end
    std=sum(Dis,1)./sum(meanSR.^2,1)/nTrial;

end