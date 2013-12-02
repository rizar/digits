function [ ] = savedata(utters, labels, path)
    for i=1:length(utters);
        filename = strcat(path, '/', int2str(labels(i)), '_', int2str(1 + floor((i - 1) / 11)), '.wav');
        audiowrite(filename, utters{i}, 11025);
    end;
end

