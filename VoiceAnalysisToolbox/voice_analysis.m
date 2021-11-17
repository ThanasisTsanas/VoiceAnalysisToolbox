function [measures_vector, measures_names, F0] = voice_analysis(data, fs, f0_alg)
%
% General call: [dysphonia_measures, dysphonia_measures_names, F0] = voice_analysis(data, fs);
%           OR: [dysphonia_measures, dysphonia_measures_names, F0] = voice_analysis('thanasis_aahh.wav');
%
% Note that some *.wav files may occasionally have a slightly different
% format than the one that Matlab supports. In that case, simply read the
% file and provide the speech signal "data" and the sampling frequency "fs"
%
%% MAIN Function of the Toolbox -- calculate various patterns from the voice signals: ONLY TESTED WITH THE SUSTAINED VOWEL /a/
%
% Main function of the "Voice Analysis Toolbox"
%
% - Function which calculates various characteristics of the signal, also
%   known as "dysphonia measures"
% - The function also computes the fundamental frequency (F0) of the signal
%
%% Inputs: data         -> Speech signal (vector) OR *.wav file
%          fs           -> Sampling frequency (Hz), not required if feeding
%                          in a *.wav file as the first input
%__________________________________________________________________________
% optional inputs:  
%          f0_alg       -> Select the algorithm for computing F0
%                          'SHRP'   - Sun's algorithm
%                          'SWIPE'  - SWIPE algorithm (A. Camacho)          
% If the user has not downloaded the external functions SHRP and SWIPE, I
% use a version of PRAAT I built in Matlab                                  [default] 
% =========================================================================
% Outputs: 
%       measures_vector -> Dysphonia measures in convenient vector format
%       measures_names  -> Corresponding names for each entry in the
%                          "measures_vector"
%               F0      -> Vector with the fundamental frequency assessment
%                          every 10 milli-seconds
% =========================================================================
%
% Part of the "Voice Analysis Toolbox"
%
% -----------------------------------------------------------------------
% Useful references:
% 
% [1] A. Tsanas: "Accurate telemonitoring of Parkinson's disease symptom
%     severity using nonlinear speech signal processing and statistical
%     machine learning", D.Phil. thesis, University of Oxford, 2012
%
% [2] A. Tsanas, M.A. Little, P.E. McSharry, L.O. Ramig: "Nonlinear speech 
%     analysis algorithms mapped to a standard metric achieve clinically 
%     useful quantification of average Parkinson’s disease symptom severity", 
%     Journal of the Royal Society Interface, Vol. 8, pp. 842-855, June 2011
%
% [3] A. Tsanas, M.A. Little, P.E. McSharry, L.O. Ramig: "New nonlinear 
%     markers and insights into speech signal degradation for effective 
%     tracking of Parkinson’s disease symptom severity", International
%     Symposium on Nonlinear Theory and its Applications (NOLTA), 
%     pp. 457-460, Krakow, Poland, 5-8 September 2010
%
% [4] A. Tsanas, M.A. Little, P.E. McSharry, L.O. Ramig: “Enhanced classical
%     dysphonia measures and sparse regression for telemonitoring of 
%     Parkinson's disease progression”, IEEE Signal Processing Society, 
%     International Conference on Acoustics, Speech and Signal Processing 
%     (ICASSP), pp. 594-597, Dallas, Texas, US, 14-19 March 2010
%
% [5] A. Tsanas, M.A. Little, C. Fox, L.O. Ramig: "Objective automatic 
%     assessment of rehabilitative speech treatment in Parkinson’s disease",
%     IEEE Transactions on Neural Systems and Rehabilitation Engineering, 
%     Vol. 22, pp. 181-190, January 2014
%
% [6] A. Tsanas: "Automatic objective biomarkers of neurodegenerative 
%     disorders using nonlinear speech signal processing tools", 8th 
%     International Workshop on Models and Analysis of Vocal Emissions for 
%     Biomedical Applications (MAVEBA), pp. 37-40, Florence, Italy, 16-18 
%     December 2013 
% -----------------------------------------------------------------------
%
% Copyright (c) Athanasios Tsanas, 2014
%
% History of modifications:
% ~~~~~~~~~~~~~~~~~~~~~~~~~
% 7 March 2014: original version of the toolbox
% 4 November 2015: updated code to deal with dual channel data
%
% ********************************************************************
% If you use this program please cite:
%
% 1) A. Tsanas, M. Little, P. McSharry and L. Ramig: "Nonlinear speech
%    analysis algorithms mapped to a standard metric achieve clinically 
%    useful quantification of average Parkinson’s disease symptom severity",
%    Journal of the Royal Society Interface, Vol. 8, pp. 842-855, June 2011
%
% 2) A. Tsanas: "Accurate telemonitoring of Parkinson's disease symptom
%    severity using nonlinear speech signal processing and statistical
%    machine learning", D.Phil. thesis, University of Oxford, 2012
%
% 3) A. Tsanas: "Automatic objective biomarkers of neurodegenerative 
%    disorders using nonlinear speech signal processing tools", 8th
%    International Workshop on Models and Analysis of Vocal Emissions for 
%    Biomedical Applications (MAVEBA), pp. 37-40, Florence, Italy, 16-18 
%    December 2013
% ********************************************************************
%
% For any question, to report bugs, or just to say this was useful, email
% tsanasthanasis@gmail.com
%
%% Parameter initialization
warning off all

addpath(genpath(pwd))

if nargin<3 || isempty(f0_alg)
    f0_alg = 'SWIPE'; 
end

if(nargin==1 || ischar(data)) % if only one input is provided
    [pathstr, name, ext] = fileparts(data);
    if(strcmp(ext, '.wav')) % a wav file was included as input; read data
        [data,fs] = wavread(data);    
    else
        error('You must provide a valid *.wav file or a vector with the speech signal and the sampling frequency!')
    end
end

% Default params for DFA
Tmax = 1000;
d = 4; % Embedding dimension
tau = 50; % Embedding delay
eta = 0.2; % RPDE close returns radius
dfa_scaling = (50:20:200)'; % DFA scaling range

f0min = 50; % min F0 possible --- Adjust depending on application!
f0max = 500; % max F0 possible --- Adjust depending on application!
flag = 1;

% take care for the case of dual channel data (potentially this is happening, if reading directly the wav file), work on the first dimension
[N,M] = size(data); if(N>M), data = data(:,1); else data = data(1,:); end

% pre-process data
data = data - mean(data);
data = data/max(abs(data));

% Use your favourite tool for estimating the fundamental frequency
switch (f0_alg)
    case 'SHRP'
        if (exist('shrp.m'))
            % Sun's implementation -> interested in F0
            F0 = shrp(data, fs, [f0min, f0max], 10);
        else
            disp('SHRP not downloaded; using alternative F0 estimator instead');
            F0 = F0_Thanasis(data,fs); % avoid problems if the user has not downloaded the SWIPE algorithm
        end
        
    case 'SWIPE'
        if (exist('swipep.m'))
            % SWIPE algorithm
            F0 = swipep(data, fs, [f0min, f0max], 0.01);
        else
            disp('SWIPE not downloaded; using alternative F0 estimator instead');
            F0 = F0_Thanasis(data,fs); % avoid problems if the user has not downloaded the SWIPE algorithm
        end
        
    otherwise
        error('You must specify an appropriate F0 estimation algorithm!')
end

if (exist('dypsa.m'))
    % Work with dypsa
    [VF_close, VF_open] = dypsa(data,fs);
    % get A0
    N=length(VF_close);
    for i=1:N-1
        A0(i) = max(abs(data(VF_open(i):VF_close(i+1))));
    end
else
    data_buffered = buffer(data,0.01*fs);
    A0 = max(data_buffered); %max value every 10 msec
end
        
data1=data; % for safety!
measures_names = []; %this will be set later!

%% Feature extraction

% jitter variants
[Jitter] = jitter_shimmer(F0);
% [Jitter2] = jitter_shimmer(1/F0); % work with pitch -- my JRSI2011 paper 
% demonstrated this does not really add additional information to jitter 
% measures when simply using the F0 contour (at least for PD applications)

% shimmer variants
[Shimmer] = jitter_shimmer(A0);

% Harmonics to Noise Ratio (HNR) and Noise to Harmonics Ratio (NHR)
[HNR,NHR] = HNRFun(data,fs);

% DFA 
if (exist('fastdfa.m'))
    dfa = fastdfa(data, dfa_scaling);
    DFA = 1/(1+exp(-dfa));
else
    DFA = NaN;
end

%RPDE
if (exist('rpde.m'))
    data_resampled = resample(data, 25000, fs);% Resample to analysis rate
    RPDE = rpde(data_resampled, d, tau, eta, Tmax); %single element vector
else
    RPDE = NaN;
end

% PPE2 (improved version compared to Max's)
F0mean_healthy_control = 120; % mean(F0); % strictly speaking, adjust this for healthy controls males = 120 and females = 190
logF0signal = log_bb(F0/F0mean_healthy_control); % log transform F0 is beneficial - see my ICASSP2010 paper, also controlling for F0 in healthy controls
ARcoef = arcov(logF0signal, 10); % identify AR coefficients
sig_filtered = filter(ARcoef, 1, logF0signal);
sig_filtered = sig_filtered(round(0.001*fs):end);
% Obtain measure of dispersion - use simple histogramic means for now
PPEd = hist(sig_filtered, linspace(min(sig_filtered), max(sig_filtered), 100));
PPE = H_entropy(PPEd/sum(PPEd))/log_bb(length(PPEd));

% Glottis open/closed quotient [Glottis Quotient (GQ) in the JRSI paper]
if (exist('dypsa.m'))
    [GQ] = glottis_quotient(VF_close,VF_open, fs, f0min, f0max, flag);
else
    GQ = NaN*ones(1,3);
end

% Glottal to Noise Excitation (GNE) related measures
[GNE] = GNE_measure(data,fs);

% Vocal Fold Excitation Ratios (VFER) in the JRSI2011 paper
if (exist('dypsa.m'))
    [VFER] = VFER_measure(data,fs);
else
    VFER = NaN*ones(1,7);
end

% Empirical Mode Decomposition Excitations Ratios (EMD-ER) in the JRSI paper
if (exist('emd.m'))
    [EMD_ER] = IMF_measure(data1);
else
    EMD_ER = NaN*ones(1,6);
end

% MFCCs
if (exist('melcepst.m'))
    mfcc = melcepst(data, fs, 'e0dD');
    MFCCs_mean = mean(mfcc);
    MFCCs_std = std(mfcc);
else
    MFCCs_mean = NaN*ones(1,42);
    MFCCs_std = NaN*ones(1,42);
end

% Wavelet measures (my NOLTA2010 paper)
[wavelet_features, wavelet_feature_names] = wavedec_features(F0);

% Summarize ALL dysphonia measures in a single vector
measures_vector = [Jitter, Shimmer, HNR, NHR, GQ, GNE, VFER, EMD_ER, MFCCs_mean, MFCCs_std, wavelet_features, PPE, DFA, RPDE];

% Get the names of the variables correctly
load ('measures_names.mat')

end % end of main function

%==========================================================================
%==========================================================================
%% Additional functions

function [measures] = jitter_shimmer(A0)

mean_Ampl = mean(A0);

% Mean absolute difference of successive cycles
measures(1) = mean(abs(diff(A0)));

% Mean absolute difference of successive cycles - expressed in percent (%)
measures(2) = 100*mean(abs(diff(A0)))/mean_Ampl;

% Perturbation quotient
[Ampl_PQ3] = perq1(A0,3);
measures(3) = Ampl_PQ3.classical_Schoentgen;
measures(4) = Ampl_PQ3.classical_Baken;
measures(5) = Ampl_PQ3.generalised_Schoentgen;

[Ampl_PQ5]=perq1(A0,5); % Use 5 cycle samples (Schoentgen)
measures(6)=Ampl_PQ5.classical_Schoentgen;
measures(7)=Ampl_PQ5.classical_Baken;
measures(8)=Ampl_PQ5.generalised_Schoentgen;

[Ampl_PQ11]=perq1(A0,11); % Use 11 cycle samples (Schoentgen)
measures(9)=Ampl_PQ11.classical_Schoentgen;
measures(10)=Ampl_PQ11.classical_Baken;
measures(11)=Ampl_PQ11.generalised_Schoentgen;

% zeroth order perturbation
measures(12) = mean(abs(A0-mean_Ampl));

% Shimmer(dB)
measures(13) = mean(20*(abs(log10((A0(1:end-1))./(A0(2:end))))));

% CV
measures(14)=mean((diff(A0)).^2)/(mean_Ampl)^2;

% TKEO
measures(15) = mean(abs(TKEO(A0)));
measures(16) = std(TKEO(A0));
Ampl_TKEO_prc = prctile(TKEO(A0),[5 25 50 75 95]);
measures(17)=Ampl_TKEO_prc(1);
measures(18)=Ampl_TKEO_prc(2);
measures(19)=Ampl_TKEO_prc(3);
measures(20)=Ampl_TKEO_prc(4);

% AM
measures(21) = (max(A0)-min(A0))/(max(A0)+min(A0));

measures(22) = Ampl_TKEO_prc(4) - Ampl_TKEO_prc(1);
end

function PQ = perq1(time_series, K)

%% Calculate the PQ using the classical PQ formula

N = length(time_series);
mean_tseries = mean(time_series);
K1=round(K/2);
K2=K-K1;
p = 5;
sum1=0;

for i = K1:N-K2
    sum1 = sum1+mean(abs([time_series(i-K2:i+K2)]-time_series(i)));
end
        
PQ.classical_Schoentgen = (sum1/(N-K+1))/(mean_tseries);

sum2=0;
for i = K1:N-K2
    sum2 = sum2+mean(abs([time_series(i-K2:i+K2)]))-time_series(i);
end
        
PQ.classical_Baken = (sum2/(N-K+1))/(mean_tseries);

% perturbation quotient of the residue
time_series=time_series(:);
sum3=0;
% calculate the AR coefficients (I use the Yule-Walker equations)
new_tseries=(time_series-mean_tseries)';
a = aryule(time_series-mean_tseries,p);

for i = 1+p:N
    sum3 = sum3+abs(sum(a.*(new_tseries(i:-1:i-p))));
end

PQ.generalised_Schoentgen = (sum3/(N-p))/(mean_tseries);
    
end

function [HNR, NHR] = HNRFun(data,fs)

f0max=500; %Hz -- max value, possibly adjust for other applications
f0min=50; %Hz
tstep=0.01*fs;
x=0.08*fs;
steps=(length(data)-x)/tstep;

for i=1:steps
    
    tseries = data(i*tstep:i*tstep+x);
    tseries = tseries-mean(tseries);
    Dwindow = hann(length(tseries));
    segment_sig = tseries.*Dwindow;
    
    %% HNR computation process
    ACF = xcorr(segment_sig,'coeff');
    ACF2 = ACF(length(segment_sig):end);
    aa=fft(segment_sig);
    aa=ifft(abs(aa).^2);
    ACF_Dwindow = xcorr(Dwindow,'coeff');
    ACF_Dwindow2 = ACF_Dwindow(length(Dwindow):end);
    bb=fft(Dwindow);
    bb=ifft(abs(bb).^2);
    ACF_signal = ACF2./ACF_Dwindow2;
    ACF_signal = ACF_signal(1:round(length(ACF_signal)/3));
    rho=aa./bb;
    rho=rho(1:length(rho)/2);
    rho=rho/max(rho);
    [rx_value,rx_index] = sort(ACF_signal,'descend');
    [d1 d2] = sort(rho, 'descend');
    low_lim=ceil(fs/f0max);  % round towards positive sample number
    up_lim=floor(fs/f0min);  % round towards negative sample number
    k=2;
    while ((rx_index(k)<low_lim) || rx_index(k)>up_lim)
        k=k+1;
    end
    
    m=2;
    while ((d2(m)<low_lim) || d2(m)>up_lim)
        m=m+1;
    end
    ll(i)=d2(m);
    mm=d2(m); 
    HNR_dB_Praat(i) = 10*log10(rho(mm)/(1-rho(mm)));
    NHR_Praat(i) = (1-rho(mm))/rho(mm);

end

%% Summarize data
HNR(1)=mean(HNR_dB_Praat);
HNR(2)=std(HNR_dB_Praat);

NHR(1)=mean(NHR_Praat);
NHR(2)=std(NHR_Praat);

end

function [GQ] = glottis_quotient(VF_close, VF_open, fs, f0min, f0max, flag)
%% Calculate the glottis quotients

cycle_open=abs(VF_open(2:end)-VF_close(1:end-1));
cycle_closed=abs(VF_open(1:end-1)-VF_close(1:end-1));

% remove erroneous cycles
if flag
    low_lim=fs/f0max; % lower limit
    up_lim=fs/f0min;  % upper limit
    N=length(cycle_open);
    for i=1:N-1
        if((cycle_open(i) > up_lim) || (cycle_open(i) < low_lim))
            cycle_open(i)=NaN;
        end
        if((cycle_closed(i) > up_lim) || (cycle_closed(i) < low_lim))
            cycle_closed(i)=NaN;
        end
    end
end

%statistics in time
prc1=prctile(cycle_open,[5 95]);
cycle_open_range_5_95_perc=prc1(2)-prc1(1);
prc2=prctile(cycle_closed,[5 95]);
cycle_closed_range_5_95_perc=prc2(2)-prc2(1);

GQ(1) = (cycle_open_range_5_95_perc/(cycle_open_range_5_95_perc+cycle_closed_range_5_95_perc));
GQ(2) = (nanstd(cycle_open));
GQ(3) = (nanstd(cycle_closed));

end

function [GNE] = GNE_measure(data, fs)

filt_order=100;
new_fs=10000;
x = 0.03*new_fs;
tstep=0.01*new_fs;
BW = 1000; % bandwidth
Fshift = 500; % shift fr   
data = resample(data, new_fs, fs); 

% cut-off1
Fc1=1:Fshift:(new_fs/2-BW-500); % not to cross Nq freq!
% cut-off2
Fc2=Fc1+BW;

for j=1:length(Fc1);
    d(j) =fdesign.bandpass('n,fc1,fc2',filt_order,Fc1(j),Fc2(j),new_fs);
    hd(j) = design(d(j));
end

steps=(length(data)-x)/tstep;

for i=1:steps+1   
    tseries = data(1+(i-1)*tstep:(i-1)*tstep+x);
    Dwindow = hann(length(tseries));
    segment_sig = tseries.*Dwindow;

    a = lpc(segment_sig,13);
    est_x = filter([0 -a(2:end)],1,segment_sig);    % Estimated signal
    e = segment_sig - est_x;
    LPE = xcorr(e,'coeff');   % LPES
    LPE=LPE(length(LPE)/2:end);

    for ii=1:length(hd)
        sigBW(:,ii)=filter(hd(ii),LPE);
        sig_TKEO(ii) = mean(TKEO(sigBW(:,ii)));
        sig_energy(ii) = mean(sigBW(:,ii)).^2; 
    end
    Hilb_tr = hilbert(sigBW);
    Hilb_env = abs(Hilb_tr);
    c = xcorr(Hilb_env);
    [cval,cidx] = max(c);
    GNEm(i) = max(cval);
    
    signal_BW_TKEO(i,:) = sig_TKEO;  
    signal_BW_energy(i,:) = sig_energy;
end

signal_BW_TKEO2 = mean(log(signal_BW_TKEO)); % used for getting the noise to signal ratio
signal_energy2 = mean(log(signal_BW_energy));

% Set outputs

GNE(1) = mean(GNEm);
GNE(2) = std(GNEm);

gnTKEO = mean(signal_BW_TKEO);
gnSEO = mean(signal_BW_energy);
GNE(3) = sum(gnTKEO(1:2))/sum(gnTKEO(end-3:end));
GNE(4) = sum(gnSEO(1:2))/sum(gnSEO(end-3:end));
GNE(5) = sum(signal_BW_TKEO2(end-3:end))/sum(signal_BW_TKEO2(1:2));
GNE(6) = sum(signal_energy2(end-3:end))/sum(signal_energy2(1:2));

end


function [VFER] = VFER_measure(data, fs)

filt_order=100;
BW = 500; % bandwidth 
Fmax = (fs/2-BW-300); % Max frequency to check 
Fshift = 500; % shift 

% get VF action
[VF_close,VF_open] = dypsa(data,fs);

% cut-off1
Fc1=1:Fshift:Fmax;
% cut-off2
Fc2=Fc1+BW;

for j=1:length(Fc1);
    d(j) = fdesign.bandpass('n,fc1,fc2',filt_order,Fc1(j),Fc2(j),fs);
    hd(j) = design(d(j));
end

for i=1:length(VF_close)-1
    tseries = data(VF_close(i):VF_close(i+1));
    Dwindow = hann(length(tseries)); %Use Hanning window
    segment_sig = tseries.*Dwindow;
    
    if (length(tseries)>50)
        for ii=1:length(hd)
            thanasis = filter(hd(ii),segment_sig);
            sigBW(:,ii) = thanasis(1:50);
            sig_TKEO(ii) = mean(TKEO(sigBW(:,ii)));
            sig_SEO(ii) = mean(sigBW(:,ii)).^2; 
        end
        Hilb_tr = hilbert(sigBW);
        Hilb_env = abs(Hilb_tr);
        c = xcorr(Hilb_env);
        [cval,cidx] = max(c);
        NEm(i) = max(cval);

        signal_BW_TKEO(i,:) = sig_TKEO;
        signal_BW_SEO(i,:) = sig_SEO;        
    end
end

signal_BW_TKEO2 = mean(log(signal_BW_TKEO)); % used for getting the noise to signal ratio

% Set outputs

VFER(1) = mean(NEm);
VFER(2) = std(NEm);
VFER(3) = -sum(NEm.*log_bb(NEm));
VFTKEO = mean(signal_BW_TKEO);
VFSEO = mean(signal_BW_SEO);
VFlog_SEO = mean(log(signal_BW_SEO));

% Get 'signal to noise' ratios
VFER(4) = sum(VFTKEO(1:5))/sum(VFTKEO(6:10));
VFER(5) = sum(VFSEO(1:5))/sum(VFSEO(6:10));
VFER(6) = sum(signal_BW_TKEO2(6:10))/sum(signal_BW_TKEO2(1:5));
VFER(7) = sum(VFlog_SEO(6:10))/sum(VFlog_SEO(1:5));

end

function [IMF] = IMF_measure(data)

%% Use classical EMD

IMF_dec = emd(data);
IMF_dec=IMF_dec';
IMF_dec2=log_bb(IMF_dec); %Log transformation
[N,M]=size(IMF_dec);

for i=1:M
    IMF_decEnergy(i) = abs(mean((IMF_dec(:,i)).^2));
    IMF_decTKEO(i) = abs(mean(TKEO(IMF_dec(:,i))));
    IMF_decEntropia(i) = abs(mean(-sum(IMF_dec(:,i).*log_bb(IMF_dec(:,i)))));
    IMF_decEnergy2(i) = abs(mean((IMF_dec2(:,i)).^2));
    IMF_decTKEO2(i) = abs(mean(TKEO(IMF_dec2(:,i))));
    IMF_decEntropia2(i) = abs(mean(-sum(IMF_dec2(:,i).*log_bb(IMF_dec2(:,i)))));
end
    
% Get 'signal to noise' ratio measures
IMF(1) = sum(IMF_decEnergy(4:end))/sum(IMF_decEnergy(1:3));
IMF(2) = sum(IMF_decTKEO(4:end))/sum(IMF_decTKEO(1:3));
IMF(3) = sum(IMF_decEntropia(4:end))/sum(IMF_decEntropia(1:3));
IMF(4) = abs(sum(IMF_decEnergy2(1:2))/sum(IMF_decEnergy2(4:end)));
IMF(5) = abs(sum(IMF_decTKEO2(1:2))/sum(IMF_decTKEO2(3:end)));
IMF(6) = sum(IMF_decEntropia2(1:2))/sum(IMF_decEntropia2(3:end));

end

function [energy] = TKEO(x)

data_length=length(x);
energy=zeros(data_length,1);

energy(1)=(x(1))^2; % first sample in the vector sequence

for n=2:data_length-1
    energy(n)=(x(n))^2-x(n-1)*x(n+1); % classical TKEO equation
end

energy(data_length)=(x(data_length))^2; % last sample in the vector sequence

end % end of TKEO function

function H = H_entropy(f)
% Calculate entropy of a discrete distribution
% Usage: H = entropy(f)
%  f - input distribution as a vector
%  H - entropy
N = length(f);
H = 0;
for j = 1:N
   H = H - f(j) * log_bb(f(j));
end

end % end of H_entropy function

function pout = log_bb(pin, method)
% Function that computes the algorithm depending on the user specified
% base; if the input probability is zero it returns zero.

if nargin<2
    method = 'Nats';
end

switch (method)
    case 'Hartson' % using log10 for the entropy computation
        log_b=@log10;
        
    case 'Nats' % using ln (natural log) for the entropy computation 
        log_b=@log;
       
    otherwise % method -> 'Bits' using log2 for the entropy computation 
        log_b=@log2;
end

if pin==0
    pout=0;
else
    pout=log_b(pin);
end

end
