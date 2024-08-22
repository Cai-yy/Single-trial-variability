% This file is used to run SNN for 1 step

function [Fr,v]=runSNN_step(Fr0,W,tau,E,N,dT)
    % ini: initial state
    % W: The connection matrix
    % E: The external input, neuron number x T number x input type
    % N: The noise (optional)
    % T: Time steps to run
    % tau: The temporal parameter of node

    % Fr: The neural activity of next step
    % v: The moving speed at this point

    if ndims(E)==3
        E=squeeze(sum(E,3)) ; % sum over different kind of input
    end

    maxSpkRate= 1; % Max spike rate
    nonlinearity   = @(x) tanh(x); % nonlinear function (membrane potential -> firing rate)
    rnl = @(x) atanh(x); % Reverse nonlinear function
    nNeuron=length(Fr0);
    
    if ~exist('N','var')
        N=zeros(nNeuron,1);
    end

    % Evolve 
    H = rnl(Fr0 / maxSpkRate) * maxSpkRate;
    JR = W*nonlinearity(H) + E + N;
    H = H + (-H + JR)/(tau/dT); % The membrane potential of next step
    Fr=nonlinearity(H / maxSpkRate) * maxSpkRate; % The Fr of next step
    v = norm(Fr-Fr0);

end