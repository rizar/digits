function [reports] = runcv(sounds_path, times)
    hmm = hmmfuncs();
    ml = mlfuncs();

    [utters, features, labels] = loaddir(sounds_path);
    reports = ml.cross_validate_times(features, labels, 4, times);
end
