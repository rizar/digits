function [functions] = mlfuncs()
    functions = struct('train', @train,...
                       'predict', @predict,...
                       'test', @test,...
                       'random_benchmark', @random_benchmark,...
                       'confmat2report', @confmat2report);
end

function [models] = train(features, labels)
    hmm = hmmfuncs();
    cs = constants();

    classes = unique(sort(labels));
    models = cell(1, length(classes));
    
    for i=1:length(classes);
        y = classes(i);
        data = features(labels == y);
        models{i} = hmm.naive_model(cs.n_states, cs.n_components, data);
        models{i} = hmm.improve_model_until(models{i}, data, cs.hmm_epsilon);
    end;
end

function [preds] = predict(models, features)
    hmm = hmmfuncs();

    likes = zeros(length(features), length(models));
    for i=1:length(models)
        likes(:, i) = cellfun(@(outputs)(hmm.sequence_log_likelihood(models{i}, outputs)), features);
    end;
    
    
    [~, preds] = max(likes, [], 2);
end

function [confmat] = test(models, features, labels) 
    classes = unique(sort(labels));
    k = length(classes);
    
    confmat = zeros(k, k);
    preds = predict(models, features);
    
    for i=1:length(preds)
        confmat(labels(i), preds(i)) = confmat(labels(i), preds(i)) + 1;
    end;
end

function [confmat] = random_benchmark(features, labels, train_percent)
    train_indices = binornd(1, train_percent, 1, length(features));
    models = train(features(train_indices == 1), labels(train_indices == 1));
    confmat = test(models, features(train_indices == 0), labels(train_indices == 0));
end
 
function [ report ] = confmat2report(confmat) 
    total = sum(confmat(1, :));
    accuracy = diag(confmat) / total;
    report = struct('confmat', confmat,...
                    'clacc', accuracy,...
                    'clerr', 1 - accuracy,...
                    'avacc', mean(accuracy),...
                    'averr', 1 - mean(accuracy));
end
