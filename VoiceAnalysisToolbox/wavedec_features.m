function [features, feature_names] = wavedec_features(data, wname, dec_levels)
%
% General call: [features, feature_names] = wavedec_features(data);
%
%% Generic function to compute features based on wavelet decomposition 
%
% - The function computes the features presented in my NOLTA (2010) paper
%   The vector output "features" can be presented to a classifier/regressor
%   in a classical supervised learning setup when the response values are
%   available, or be used for clustering applications.
%
% Inputs:  data         -> in principle, any time series. In my NOLTA 2010
%                          I processed the raw time series to extract F0, 
%                          and then applied this function
%__________________________________________________________________________
% optional inputs:  
%          wname        -> the wavelet family to be use                     [default 'db8']
%          dec_levels   -> number of decomposition levels                   [default 10]
% =========================================================================
% Outputs: features     -> features in vector format
%         feature_names -> corresponding feature names
% =========================================================================
%
% -----------------------------------------------------------------------
% Useful references: look in each function for the appropriate reference
% 
% (1) A. Tsanas: "Accurate telemonitoring of Parkinson's disease symptom
%     severity using nonlinear speech signal processing and statistical
%     machine learning", D.Phil. thesis, University of Oxford, 2012
%
% (2) A. Tsanas, M.A. Little, P.E. McSharry, L.O. Ramig: "New nonlinear 
%     markers and insights into speech signal degradation for effective 
%     tracking of Parkinson’s disease symptom severity", International
%     Symposium on Nonlinear Theory and its Applications (NOLTA), 
%     pp. 457-460, Krakow, Poland, 5-8 September 2010
% -----------------------------------------------------------------------
%
% Last modified on 16 February 2014
%
% Copyright (c) Athanasios Tsanas, 2014
%
% ********************************************************************
% If you use this program please cite:
%
%     A. Tsanas, M.A. Little, P.E. McSharry, L.O. Ramig: "New nonlinear 
%     markers and insights into speech signal degradation for effective 
%     tracking of Parkinson’s disease symptom severity", International
%     Symposium on Nonlinear Theory and its Applications (NOLTA), 
%     pp. 457-460, Krakow, Poland, 5-8 September 2010
% ********************************************************************
%
% For any question, to report bugs, or just to say this was useful, email
% tsanasthanasis@gmail.com

%% Find some internal parameters before the main function

if nargin<3 || isempty(dec_levels);
    dec_levels = 10; % the user can use more levels; I used 10 in the NOLTA paper
end

if nargin<2 || isempty(wname);
    wname = 'db8'; % experiment with the wavelet base - e.g. sym4 ->NOLTA
end

% ************** Wavelet decomposition ****************
[C, L] = wavedec(data, dec_levels, wname); % compute decomposition

[F0c.Ea, F0c.Ed] = wenergy(C, L);
% Expand the discrete wavelet coefficients
for k = 1:dec_levels
    d = detcoef(C,L,k); % Detail coefficients in levels 1...dec_levels
    F0c.det_entropy_shannon(k) = wentropy(d, 'shannon');
    F0c.det_entropy_log(k) = wentropy(d, 'log energy');
    F0c.det_TKEO_mean(k) = mean(TKEO(d));
    F0c.det_TKEO_std(k) = std(TKEO(d));
end

for k = 1:dec_levels
    a = appcoef(C,L,wname, k); % Approximation coefficients in levels 1...dec_levels
    F0c.app_entropy_shannon(k) = wentropy(a, 'shannon');
    F0c.app_entropy_log(k) = wentropy(a, 'log energy');
    F0c.app_det_TKEO_mean(k) = mean(TKEO(a));
    F0c.app_TKEO_std(k) = std(TKEO(a));    
end

% Now, repeat the decomposition process using the log-transformed F0 values
% (this has been shown to be beneficial in some settings, e.g. in my ICASSP
% 2010 paper). Reason: transformation of the F0 towards a more normal
% distribution

Ldata = log(data);
clear C L d a
[C, L] = wavedec(Ldata, dec_levels, wname); % compute decomposition

[F0c.Ea2, F0c.Ed2] = wenergy(C, L);
% Expand the discrete wavelet coefficients
for k = 1:dec_levels
    d = detcoef(C,L,k);
    F0c.det_LT_entropy_shannon(k) = wentropy(d, 'shannon');
    F0c.det_LT_entropy_log(k) = wentropy(d, 'log energy');
    F0c.det_LT_TKEO_mean(k) = mean(TKEO(d));
    F0c.det_LT_TKEO_std(k) = std(TKEO(d));
end

for k = 1:dec_levels    
    a = appcoef(C,L,wname,k);
    F0c.app_LT_entropy_shannon(k) = wentropy(a, 'shannon');
    F0c.app_LT_entropy_log(k) = wentropy(a, 'log energy');
    F0c.app_LT_TKEO_mean(k) = mean(TKEO(a));
    F0c.app_LT_TKEO_std(k) = std(TKEO(a));       
end

%% Determine feature vector, get the measures in vector form

% This vector can be used as input into regression or classification learners
Measures = F0c;
Measures_names = fieldnames(Measures); % Measures is a struct with all the data in sub-structs or doubles
counter = 0;
for i=1:length(Measures_names)
    str = char(Measures_names(i)); % start accessing the sub-structs or doubles
    substr = Measures.(str); % contents of 'str'
    if (isstruct(substr))
        substruct_names = fieldnames(substr);
        for j=1:length(substruct_names) % this for loop takes features in those cases that encounters scalar values, otherwise discards them
            if(isscalar(substr.(char(substruct_names(j)))))
                counter = counter+1;
                measures_vector.features(1, counter) = substr.(char(substruct_names(j)));
                measures_vector.names{1, counter} = ([char(str) '->' char(char(substruct_names(j)))]);
            end
        end    
    elseif(numel(substr)>1) % we are dealing with a vector here [i.e. case b)]
        for j=1:length(substr)
            counter = counter+1;
            measures_vector.features(counter) = substr(j);
            % get output in the form "Main_string->sub-string"
            measures_vector.names{1, counter} = ([char(str) '_' num2str(j) '_coef']);
        end
    else
        counter = counter+1;
        measures_vector.features(counter) = substr;
        measures_vector.names(1, counter) = Measures_names(i);
    end
end 

features = measures_vector.features;
feature_names = measures_vector.names;

end % end of main function

function [energy] = TKEO(x)
% Aim: compute the nonlinear energy operator of a vector x

% Data analysis 

data_length=length(x);
energy=zeros(data_length,1);

energy(1)=(x(1))^2; % first sample in the vector sequence

for n=2:data_length-1
    energy(n)=(x(n))^2-x(n-1)*x(n+1); % classical TKEO equation
end

energy(data_length)=(x(data_length))^2; % last sample in the vector sequence

end