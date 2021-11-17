function [X, measures_names, F0, wav_files] = batch_speech_signal_analysis()

folder_name = uigetdir; cd(folder_name)
candidates = dir('*.wav'); wav_files = (sort(extractfield(candidates, 'name')))';

X = []; measures_names = []; F0 = []; 
h = waitbar(0,'Processing files...');
L = length(wav_files);
for i = 1:L
    [X(i,:), measures_names, F0{i}] = voice_analysis(wav_files{i});
%     fft(1:10000000);
    waitbar(i/L)
end

%% Save data in convenient format in Excel and binary format
data_file_Excel = ['Batch_voice_', date, '.xlsx'];
data_file_binary = ['Batch_voice_', date, '.dat'];

% write data in binary format
writetable(T, data_file_binary,'WriteRowNames',true);

% Get the data in convenient cell format and save it in Excel
measures_names2 = ['Speech files/Features', measures_names];
% [~, X2] = present_mean_std(X);
X2 = [wav_files, num2cell(X)];
Data_table = [measures_names2; X2]; 
xlswrite(data_file_Excel, Data_table, 'Sheet1');

% get the data in Table format in Matlab, and then save in Excel
measures_names2 = strrep(measures_names,'->', '__'); measures_names2 = strrep(measures_names2,' ', '_'); measures_names2 = strrep(measures_names2, '-', '_');
T = cell2table(num2cell(X), 'RowNames', wav_files, 'VariableNames', measures_names2); 
% write data in Excel
writetable(T, data_file_Excel, 'WriteRowNames',true, 'WriteVariableNames',true);

