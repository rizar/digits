function [functions] = mlfuncs()
    functions = struct('normalizing_transform', @normalizing_transform,...
                       'apply_transform', @apply_transform,...
                       'train', @train,...
                       'predict', @predict,...
                       'test', @test,...
                       'random_benchmark', @random_benchmark,...
                       'confmat2report', @confmat2report,...
                       'cross_validate', @cross_validate,...
                       'cross_validate_times', @cross_validate_times);
end

function [transform] = identity(features)
    n_features = size(features{1}, 1);
    transform = struct();
    transform.translate = zeros(n_features, 1);
    transform.scale = diag(ones(n_features, 1));
end

function [transform] = normalizing_transform(features)
    data = [features{:}];

    transform = struct();
    transform.translate = -mean(data, 2);
    transform.scale = diag(1 ./ std(data, 0, 2));
end

function [result] = apply_transform(t, features)
    result = cellfun(@(outputs)(...
            t.scale * (outputs + repmat(t.translate, 1, size(outputs, 2)))...
        ), features, 'UniformOutput', false);
end

function [classifier] = train(features, labels)
    hmm = hmmfuncs();
    cs = constants();

    classes = unique(sort(labels));
    models = cell(1, length(classes));
    
    if cs.do_normalize_features == 1;
        transform = normalizing_transform(features);
    else;
        transform = identity(features);
    end;
        
    normalized = apply_transform(transform, features);
    
    for i=1:length(classes);
        y = classes(i);
        data = normalized(labels == y);
        models{i} = hmm.naive_model(cs.n_states, cs.n_components, data);
        models{i} = hmm.improve_model_until(models{i}, data, cs.hmm_epsilon);
    end;
    
    classifier = struct();
    classifier.models = models;
    classifier.transform = transform;
end

function [preds] = predict(classifier, features)
    hmm = hmmfuncs();
    
    n_models = length(classifier.models);

    likes = zeros(length(features), n_models);
    for i=1:n_models;
        likes(:, i) = cellfun(@(outputs)(...
            hmm.max_log_likelihood(classifier.models{i}, outputs)...
            ), apply_transform(classifier.transform, features));
    end;
        
    [~, preds] = max(likes, [], 2);
end

function [confmat] = test(models, features, labels) 
    cs = constants();
    classes = unique(sort(labels));
    
    confmat = zeros(cs.n_classes, cs.n_classes);
    preds = predict(models, features);
    
    for i=1:length(preds)
        confmat(labels(i), classes(preds(i))) = confmat(labels(i), classes(preds(i))) + 1;
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
             
    % Train and test
    rng(seeds(2));
    
    tic;
    confmat_train = zeros(cs.n_classes);
    confmat_test = zeros(cs.n_classes);
    for i=1:k;
        models = train(features(train_masks(i, :) == 1), labels(train_masks(i, :) == 1));
        confmat_train = confmat_train + test(models, features(train_masks(i, :) == 1), labels(train_masks(i, :) == 1));
        confmat_test = confmat_test + test(models, features(train_masks(i, :) == 0), labels(train_masks(i, :) == 0));
    end;
    
    report = struct();
    report.train = confmat2report(confmat_train);
    report.test = confmat2report(confmat_test);
end

function [reports] = cross_validate_times(features, labels, k, times)
    constants()

    reports = struct('train', [], 'test', []);
    for t=1:times;
        reports(t) = cross_validate(features, labels, k, [t t]);
    end;
end

function [report] = confmat2report(confmat) 
    total = sum(confmat, 2);
    accuracy = diag(confmat) ./ total;
    average_accuracy = sum(diag(confmat)) / sum(total);
    report = struct('confmat', confmat,...
                    'clacc', accuracy,...
                    'clerr', 1 - accuracy,...
                    'avacc', average_accuracy,...
                    'averr', 1 - average_accuracy);
end