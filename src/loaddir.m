function [ uttrs, labels ] = loaddir(path)
    cs = constants();

    files = dir(path);
    files = files(3:length(files));
    
    uttrs = cell(cs.n_classes * length(files), 1);
    labels = zeros(length(uttrs), 1);
    current = 0;
    
    for i=1:length(files);
        sig = audioread(strcat('sounds/', char(files(i).name)));
        try;
            digits = cut(sig);
            for c=1:cs.n_classes;
                uttrs{current + c} = digits{c};
                labels(current + c) = c;
            end;
            current = current + cs.n_classes;
        catch err;
            sprintf('%s: %s', files(i).name, err.message)
        end;
    end;
    
    uttrs = uttrs(1:current);
    labels = labels(1:current);
end

