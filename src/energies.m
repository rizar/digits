function [ res ] = energies(sig)
    cs = constants();
    
    n_frames = floor(length(sig) / cs.window_size);
        
    % calculate frame energies
    res = zeros(1, n_frames);
    for i=1:n_frames;
        start = cs.window_size * (i - 1) + 1;
        finish = cs.window_size * i;
        res(i) = norm(sig(start:finish)) ^ 2;
    end;

    % tmp = sort(res);
    % normalizer = 1 / mean(tmp(ceil(0.9 * length(tmp)):length(tmp)));
    % res = res * normalizer;
end

