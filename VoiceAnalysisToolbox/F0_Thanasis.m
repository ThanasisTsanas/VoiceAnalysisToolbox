function [F0,f0_times] = F0_Thanasis(data, fs, x, tstep, f0min, f0max)
%
% General call: [F0,f0_times] = F0_Thanasis(data,fs);
%
%% Function to calculate the Fundamental Frequency F0 using the Praat algorithm 
%  (note that this function does not use dynamic programming at the end to refine the F0 estimates, which is what Praat's final post-processing step does)
%
% Inputs:  data            -> raw signal data (vector)
%          fs              -> Sampling frequency (Hz)
%__________________________________________________________________________
% optional inputs:         
%
%          x               -> length of time series (sec) at each step     [default 0.04*fs] 
%          tstep           -> time step within the algorithm (sec)         [default 0.01*fs]
%          f0min           -> minimum F0                                   [default 50]
%          f0max           -> maximum F0                                   [default 500]
% =========================================================================
% Outputs: F0              -> Fundamental frequency vector
%          f0_times        -> Instances where F0 is evaluated
% =========================================================================
%
% Part of the "Voice Analysis Toolbox" by A. Tsanas
%
% -----------------------------------------------------------------------
% Useful references:
% 
% 1) A. Tsanas: "Accurate telemonitoring of Parkinson's disease symptom
%    severity using nonlinear speech signal processing and statistical
%    machine learning", D.Phil. thesis, University of Oxford, 2012
% -----------------------------------------------------------------------
%
% Last modified on 7 March 2014
%
% Copyright (c) Athanasios Tsanas
%
% ********************************************************************
% If you use this program please cite at least one of the following:
%
% 1) A. Tsanas: "Accurate telemonitoring of Parkinson's disease symptom
%    severity using nonlinear speech signal processing and statistical
%    machine learning", D.Phil. thesis, University of Oxford, 2012
%
% 2) A. Tsanas: "Automatic objective biomarkers of neurodegenerative 
%    disorders using nonlinear speech signal processing tools", 8th
%    International Workshop on Models and Analysis of Vocal Emissions for 
%    Biomedical Applications (MAVEBA), pp. 37-40, Florence, Italy, 16-18 
%    December 2013
% ********************************************************************
%
% For any question, to report bugs, or just to say this was useful, email
% tsanasthanasis@gmail.com

%warning off all % disable some annoying warnings appearing all the time 

%% Check inputs and use defaults

if nargin<3 || isempty(x)
    %Determine length of time series to be used at each step 
    x=0.04*fs; % 40 msec according to Boersma
end

if nargin<4 || isempty(tstep)
    % use the standard time-step
    tstep=0.01*fs;
end

if nargin<5 || isempty(f0min)
    % use the standard minimum F0
    f0min=50; %Hz
end

if nargin<6 || isempty(f0max)
    % use the standard maximum F0
    f0max=500; %Hz
end

%% Data processing

steps=round((length(data)-x)/tstep);
f0_Praat = zeros(steps+1,1);
f0_times = ((x/(2*fs)):tstep/fs:(steps+2)*tstep/fs)'; 
f0_times = f0_times .*1000; % msec -> sec

for i=1:steps
    
    % define time-series frame
    tseries = data(1+(i-1)*tstep:(i-1)*tstep+x);
    tseries = tseries-mean(tseries);
    Dwindow = gausswin(length(tseries));
    segment_sig = tseries.*Dwindow;

    %% F0 computation process
    
    % Calculate *signal* auto-correlation
    ACF = xcorr(segment_sig,'coeff');
    ACF2 = ACF(length(segment_sig):end);
    aa=fft(segment_sig);
    aa=ifft(abs(aa).^2);
    aa=aa/max(aa); %normalize autocorrelation

    % Calculate *window* auto-correlation, and use only points following 0
    ACF_Dwindow = xcorr(Dwindow,'coeff');
    ACF_Dwindow2 = ACF_Dwindow(length(Dwindow):end);
    
    %Praat
    bb=fft(Dwindow);
    bb=ifft(abs(bb).^2);
    bb=bb/max(bb); % normalize autocorrelation
    ACF_signal = ACF2./ACF_Dwindow2;
    ACF_signal = ACF_signal(1:round(length(ACF_signal)/3));
    
    rho=aa./bb;
    rho=rho(1:round(length(rho)/2));
    rho=rho/max(rho); % since max(rho)=1 this is not really needed

    % sort the autocorrelation values and choose the one with the max value
    [rx_value,rx_index] = sort(ACF_signal,'descend');
    
    %Boersma
    [d1 d2] = sort(rho, 'descend');

    % Guard against illusion harmonics
    low_lim=ceil(fs/f0max);  % round towards positive sample number
    up_lim=floor(fs/f0min);  % round towards negative sample number
    k=2;
    while ((rx_index(k)<low_lim) || rx_index(k)>up_lim)
        k=k+1;
    end
    kk(i)=rx_index(k);

    m=2;
    while ((d2(m)<low_lim) || d2(m)>up_lim)
        m=m+1;
    end
          
    %approach with interpolation
    dt=1/fs;
    mm=d2(m);   
    if (mm>low_lim && mm<up_lim) % algorithm runs normal according to Boersma
        tmax1 = 0.5*(rho(mm+1)-rho(mm-1));
        tmax2 = 2*rho(mm) - rho(mm-1) - rho(mm+1);
        tmax = dt*(mm + tmax1/tmax2);
    else
        % guard against limit findings -> modify Boersma's equations
        tmax = mm*dt; %that's actually equivalent to the crude approach!
    end
    
    f0_Praat1(i) = 1/tmax;
end

%% Summarize data

F0 = f0_Praat1(:);
