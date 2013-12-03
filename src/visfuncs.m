function [ functions ] = visfuncs()
    functions = struct('show_durations', @show_durations,...
                       'plot_digit', @plot_digit);
end

function [] = plot_digit(features, labels, digits)
    ml = mlfuncs();
    tr_norm = ml.normalizing_transform(features);
    fs_norm = ml.apply_transform(tr_norm, features);
    tr_pca = ml.pca_transform(fs_norm);
    fs_pca = ml.apply_transform(tr_pca, fs_norm);
    paths = fs_pca(labels == digits);
    
    figure;
    %hold on;
    for i=1:length(paths);
        p = paths{i};
        plot(p(1, :), p(2, :));
    end;
end

