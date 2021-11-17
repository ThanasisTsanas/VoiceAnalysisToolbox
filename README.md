# VoiceAnalysisToolbox
Voice Analysis Toolbox

Toolbox to analyze sustained vowel phonations. This is largely based on my PhD work, and has been extensively tested on processing sustained vowel /a/ phonations.

Simply run "voice_analysis_toolbox.m". You should provide either (a) a '*.wav' file or (b) the sustained vowel signal 'data' and the sampling frequency 'fs' as inputs into the function. Ideally, high quality speech signals should be used (e.g. with sampling frequency of 24 kHz). Indicative use:
[features, feature_names] = voice_analysis.toolbox('thanasis_aahh.wav')

**Downloading necessary third party software & SETUP**
Before using the Voice Analysis Toolbox, you will need to download software from other researchers (third party software). I apologise in advance for complicating the user‟s task, but this is an easier way to adhere to different licence issues. To facilitate this process, I have included the function “setup_voice_analysis_toolbox” which the user should run on a machine which is connected on the Internet to download the required files. If the authors of the third party software move their functions, the user may have to manually download the required functions from the authors‟/developers‟ websites.

**More details about the algorithms used in the toolbox**
The toolbox contains functions developed over a series of papers in the last few years. The main research studies for which this toolbox was developed appear below in order of decreasing importance:
[1] A. Tsanas, M.A. Little, P.E. McSharry, L.O. Ramig: “Nonlinear speech analysis algorithms mapped to a standard metric achieve clinically useful quantification of average Parkinson‟s disease symptom severity”, Journal of the Royal Society Interface, Vol. 8, pp. 842-855, 2011
[2] A. Tsanas, Accurate telemonitoring of Parkinson’s disease symptom severity using nonlinear speech signal processing and statistical machine learning, D.Phil. (Ph.D.) thesis, University of Oxford, UK, 2012
[3] A. Tsanas, M.A. Little, P.E. McSharry, L.O. Ramig: “New nonlinear markers and insights into speech signal degradation for effective tracking of Parkinson‟s disease symptom severity”, International Symposium on Nonlinear Theory and its Applications (NOLTA), pp. 457-460, Krakow, Poland, 5-8 September 2010
[4] A. Tsanas, M.A. Little, P.E. McSharry, L.O. Ramig: “Enhanced classical dysphonia measures and sparse regression for telemonitoring of Parkinson's disease progression”, IEEE Signal Processing Society, International Conference on Acoustics, Speech and Signal Processing (ICASSP), pp. 594-597, Dallas, Texas, US, 14-19 March 2010
[5] A. Tsanas: “Automatic objective biomarkers of neurodegenerative disorders using nonlinear speech signal processing tools”, 8th International Workshop on Models and Analysis of Vocal Emissions for Biomedical Applications (MAVEBA), pp. 37-40, Florence, Italy, 16-18 December 2013

**Citation request**
If you use this toolbox in your research, please include the following citations:
[1] A. Tsanas, M.A. Little, P.E. McSharry, L.O. Ramig: “Nonlinear speech analysis algorithms mapped to a standard metric achieve clinically useful quantification of average Parkinson‟s disease symptom severity”, Journal of the Royal Society Interface, Vol. 8, pp. 842-855, 2011
[2] A. Tsanas, Accurate telemonitoring of Parkinson’s disease symptom severity using nonlinear speech signal processing and statistical machine learning, D.Phil. (Ph.D.) thesis, University of Oxford, UK, 2012
[3] A. Tsanas: “Automatic objective biomarkers of neurodegenerative disorders using nonlinear speech signal processing tools”, 8th International Workshop on Models and Analysis of Vocal Emissions for Biomedical Applications (MAVEBA), pp. 37-40, Florence, Italy, 16-18 December 2013
