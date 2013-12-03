function [functions] = utilfuncs();
    functions = struct()
    functions.padcat = @padcat;
end

function [res] = padcat(varargin)
    va = varargin;
    max_length = max(cellfun(@(a)(length(a)), va));
    res = zeros(max_length, length(va)); 
    for i=1:length(va)
        res(1:length(va{i}), i) = va{i};
    end;
end

