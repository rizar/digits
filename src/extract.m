function [features] = extract(signal, use_filtering, all_cepstrum)
    n = size(signal, 1);
    
    % filtering and windowing
    if use_filtering == 1;
        filtered_signal = filter([1 -0.97], 1, signal);
    else;
        filtered_signal = signal;
    end;
    % fs = signal;
    windowed_signal = bsxfun(@times, filtered_signal, hamming(n));
        
    % calculate sqrt(power) spectrum
    magnitudes = abs(fft(windowed_signal));
    % throwing away the symmetric part
    magnitudes = magnitudes(1:n/2, :);
    % spectrum averaging
    magnitudes = mean(magnitudes, 2);
    
    % calculate cepstrum
    logarithms = log(magnitudes);
    cepstrum = abs(fft(logarithms));
    % throwing away the symmetric part
    cepstrum = cepstrum(1:n/4, :);
    
    % take some first cepstrum coefficients as features
    params = constants();
    if nargin < 3; 
        features = cepstrum(2:params.n_features + 1);
    else;
        features = cepstrum;
    end
end
