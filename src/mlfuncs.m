function [functions] = mlfuncs()
    functions = struct('train', @train,...
                       'predict', @predict,...
                       'test', @test,...
                       'random_benchmark', @random_benchmark,...
                       'confmat2report', @confmat2report,...
                       'cross_validate', @cross_validate);
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
        likes(:, i) = cellfun(@(outputs)(hmm.max_log_likelihood(models{i}, outputs)), features);
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

function [report] = random_benchmark(features, labels, train_percent, seeds)
    tic;
    
    rng(seeds(1));

    cs = constants();
    from_each = floor(length(features) * train_percent / cs.n_classes);
    
    train_mask = zeros(1, length(features));
    for y=1:cs.n_classes;
        y_indices = find(labels == y);
        p = randperm(length(y_indices));
        train_mask(y_indices(p(1:from_each))) = 1;
    end;
    sprintf('train size %d', sum(train_mask))
          
    rng(seeds(2));
    
    models = train(features(train_mask == 1), labels(train_mask == 1));
    confmat_test = test(models, features(train_mask == 0), labels(train_mask == 0));
    confmat_train = test(models, features(train_mask == 1), labels(train_mask == 1));
    
    report = struct();
    report.models = models;
    report.confmat_train = confmat_train;
    report.confmat_test = confmat_test;
    report.train_mask = train_mask;
    
    toc
end
 
function [report] = cross_validate(features, labels, k, seeds)
    cs = constants();

    % Generate masks
    rng(seeds(1));
   
    train_masks = zeros(k, length(features));
    for y=1:cs.n_classes;
        y_indices = find(labels == y);
        p = randperm(length(y_indices));
        for i=1:k;
            tmp = (1:length(y_indices)) / length(y_indices);
            test_indices = (i - 1) / k < tmp & tmp <= i / k;
            train_masks(i, y_indices(p(~test_indices))) = 1;
        end;
    end;
    assert(all(sum(train_masks, 1) == (k - 1) * ones(1, length(features))));
    sprintf(num2str(sum(train_masks, 2)'))
             
    % Train and test
    rng(seeds(2));
    
    tic;
    confmat = zeros(cs.n_classes);
    for i=1:k;
        models = train(features(train_masks(i, :) == 1), labels(train_masks(i, :) == 1));
        confmat = confmat + test(models, features(train_masks(i, :) == 0), labels(train_masks(i, :) == 0));
        toc
    end;
    
    report = confmat;
end

function [ report ] = confmat2report(confmat) 
    total = sum(confmat, 2);
    accuracy = diag(confmat) ./ total;
    average_accuracy = sum(diag(confmat)) / sum(total);
    report = struct('confmat', confmat,...
                    'clacc', accuracy,...
                    'clerr', 1 - accuracy,...
                    'avacc', average_accuracy,...
                    'averr', 1 - average_accuracy);
end
