function [features] = featurize(utterance)
    cs = constants();
    shift = cs.window_size / 2;
    features = zeros(cs.n_features, length(utterance) / shift);
    for i=1:size(features, 2)
        features(:, i) = extract(utterance(shift * (i - 1) + 1:shift * i), 1, 0);
    end;
end
