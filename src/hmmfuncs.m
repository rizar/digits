function [functions] = hmmfuncs()
    functions = struct('create_model', @create_model,...
                       'check_model', @check_model,...
                       'generate_sequence', @generate_sequence,...
                       'generate_n_sequences', @generate_n_sequences,...
                       'generate_models', @generate_models,...
                       'naive_model', @naive_model,...
                       'mixture_likelihood', @mixture_likelihood,...
                       'state_likelihood', @state_likelihood,...
                       'sequence_log_likelihood', @sequence_log_likelihood,...
                       'n_sequences_log_likelihood', @n_sequences_log_likelihood,...
                       'forward_procedure', @forward_procedure,...
                       'backward_procedure', @backward_procedure,...
                       'improve_model', @improve_model,...
                       'improve_model_until', @improve_model_until,...
                       'draw_model', @draw_model,...
                       'draw_sequence', @draw_sequence,...
                       'draw_component_part', @draw_component_part);
end

function [params] = internal_params()
    params = struct('transition_smoother', 0.01,...
                    'mixture_smoother', 0.01,...
                    'variance_smoother', 0.01);
end

function [model] = create_model(A, MX, MN, VS)
    model = struct('transitions', A,...
                   'mixtures', MX,...
                   'means', MN,...
                   'variances', VS);
end                   
                    
function [] = check_model(model)
    n_states = size(model.transitions, 1);
    if norm(sum(model.transitions, 2) - ones(n_states, 1)) > 1e-6;
        error('invalid transition matrix');
    end;
    if norm(sum(model.mixtures, 2) - ones(n_states, 1)) > 1e-6;
        error('invalid mixture');
    end;
end

function [outputs, states] = generate_sequence(ps)
    n_states = size(ps.transitions, 1);
    n_features = size(ps.means, 3);
    
    states = zeros(1, 1000);
    outputs = zeros(n_features, 1000);
    
    i = 1;
    states(i) = 1;
    
    while 1
        s = states(i);
        
        % generate output
        comp = find(mnrnd(1, ps.mixtures(s, :)) > 0);
        mu = reshape(ps.means(s, comp, :), 1, n_features);
        sigma = diag(reshape(ps.variances(s, comp, :), 1, n_features));
        outputs(:, i) = mvnrnd(mu, sigma);
        
        if s == n_states;
            break;
        end;
        
        % do transition
        states(i + 1) = find(mnrnd(1, ps.transitions(s, :)) > 0);
        i = i + 1;
    end;
    
    states = states(1:i);
    outputs = outputs(:, 1:i);
end

function [outputs] = generate_n_sequences(n, ps)
    outputs = cell(n, 1);
    for i=1:n
        outputs{i} = generate_sequence(ps);
    end;  
end

function [models] = generate_models(n_models, n_states, n_components, n_features)
    models = cell(1, n_models);
    for model_index=1:n_models
        % generate upper-diagonal transition matrix
        A = zeros(n_states, n_states);
        for i=1:n_states;
            A(i, i) = (rand(1) + 3) / 4;
            if i < n_states
                xs = rand(1, n_states - i);
                A(i, i+1:n_states) = xs ./ sum(xs) .* (1 - A(i, i));
            end;
        end;
        A(n_states, n_states) = 1;
   
        % generate mixture matrix
        MX = zeros(n_states, n_components);
        for i=1:n_states;
            xs = rand(1, n_components);
            MX(i, :) = xs ./ sum(xs);
        end;
        
        % generate means matrix
        MN = zeros(n_states, n_components, n_features);
        for i=1:n_states;
            for j=1:n_components;
                MN(i, j, :) = 10 * rand(1, n_features);
            end;
        end;
        
        % generate variance matrices
        VS = zeros(n_states, n_components, n_features);
        for i=1:n_states;
            for j=1:n_components;
                VS(i, j, :) = 3 * rand(1, n_features);
            end;
        end;
        
        models{model_index} = struct('transitions', A,...
                                     'mixtures', MX,...
                                     'means', MN,...
                                     'variances', VS);
    end;    
end

function [res] = likelihoods(means, variances, output)
    n_components = size(means, 1);
    
    res = zeros(1, n_components);
    for i=1:n_components;
        centered = output - means(i, :);
        dist2 = sum(centered .^ 2 ./ variances(i, :));
        res(i) = exp(-0.5 * dist2) / sqrt(2 * pi * prod(variances(i, :)));
    end;
    assert(all(-1e-7 <= res));
 end

function [res] = mixture_likelihood(probs, means, variances, output)
    likes = likelihoods(means, variances, output);
    res = sum(probs .* likes);
end

function [res] = state_likelihood(model, state, output)
    res = mixture_likelihood(model.mixtures(state, :),...
                squeeze(model.means(state, :, :)),...
                squeeze(model.variances(state, :, :)),...
                output);
end

function [res] = state_component_likelihoods(model, state, output)
    res = likelihoods(squeeze(model.means(state, :, :)),...
                squeeze(model.variances(state, :, :)),...
                output);
end

function [prefix_probs, scalers] = forward_procedure(model, outputs)
    len = size(outputs, 2);
    n_states = size(model.transitions, 1);
    
    prefix_probs = zeros(len, n_states);    
    prefix_probs(1, 1) = 1;
    scalers = zeros(len, 1);
    scalers(1) = 1 / state_likelihood(model, 1, outputs(:, 1)');
    
    for t=2:len;
        for j=1:n_states;
            for i=1:n_states;
                prefix_probs(t, j) = prefix_probs(t, j) +...
                    prefix_probs(t - 1, i) * model.transitions(i, j);
            end;
            prefix_probs(t, j) = prefix_probs(t, j) * state_likelihood(model, j, outputs(:, t)');
        end;
        
        scalers(t) = 1 / sum(prefix_probs(t, :));
        prefix_probs(t, :) = prefix_probs(t, :) .* scalers(t);
    end;
end

function [suffix_probs] = backward_procedure(model, outputs, scalers)
    len = size(outputs, 2);
    n_states = size(model.transitions, 1);
    
    suffix_probs = zeros(len, n_states);
    suffix_probs(len, :) = scalers(len);
    
    for t=len-1:-1:1;
        for i=1:n_states;
            for j=1:n_states;
                suffix_probs(t, i) = suffix_probs(t, i) +...
                    model.transitions(i, j) * state_likelihood(model, j, outputs(:, t + 1)')...
                    * suffix_probs(t + 1, j);
            end;
        end;
        suffix_probs(t, :) = suffix_probs(t, :) .* scalers(t);            
    end;    
end

function [res] = sequence_log_likelihood(model, outputs)
    [prefix_probs, scalers] = forward_procedure(model, outputs);
    total = sum(prefix_probs(size(outputs, 2), :));
    res = log(total) - sum(log(scalers));
end

function [res] = n_sequences_log_likelihood(model, all_outputs)
    res = 0;
    for i = 1:length(all_outputs);
        res = res + sequence_log_likelihood(model, all_outputs{i});
    end;
end

function [result] = naive_model(n_states, n_components, all_outputs)
    average_len = length([all_outputs{:}]) / n_states;
    n_features = size(all_outputs{1}, 1);
    
    A = zeros(n_states, n_states);
    for i=1:n_states;
        A(i, i) = (average_len - 1) / average_len;
        A(i, i+1:n_states) = (1 - A(i, i)) / (n_states - i);
    end;
    A(n_states, n_states) = 1;
    
    MN = zeros(n_states, n_components, n_features);
    VS = zeros(n_states, n_components, n_features);
    
    for i=1:n_states;
        take = cell(length(all_outputs));
        for k=1:length(all_outputs);
            len = size(all_outputs{k}, 2);
            mask = false(1, len);
            start = floor(len * (i - 1) / n_states) + 1;
            finish = floor(len * i / n_states);
            mask(start:finish) = true;
            take{k} = mask;
        end;
        all_data = [all_outputs{:}];
        big_mask = [take{:}];
        state_data = all_data(:, big_mask > 0);
        
        center = mean(state_data, 2);
        variances = var(state_data, 1, 2);
        
        means = mvnrnd(center, diag(variances / 4), n_components);
        
        MN(i, :, :) = means;
        VS(i, :, :) = repmat(variances', n_components, 1);
    end;
    
    result = create_model(A, ones(n_states, n_components) ./ n_components, MN, VS);
    check_model(result);
end

function [result] = improve_model(model, all_outputs)
    n_states = size(model.transitions, 1);
    n_components = size(model.mixtures, 2);
    n_features = size(model.means, 3);
    
    ip = internal_params();

    state_expect = zeros(n_states, 1);
    last_expect = zeros(n_states, 1);
    transition_expect = zeros(n_states, n_states);
    
    state_component_expect = zeros(n_states, n_components);
    state_component_feature_sum = zeros(n_states, n_components, n_features);
    state_component_feature_square_sum = zeros(n_states, n_components, n_features);
    
    for k=1:size(all_outputs, 1);
        outputs = all_outputs{k};
        [prefix_probs, scalers] = forward_procedure(model, outputs);
        suffix_probs = backward_procedure(model, outputs, scalers);
        
        for t=1:size(outputs, 2);
            for i=1:n_states;
                state_prob = prefix_probs(t, i) * suffix_probs(t, i) / scalers(t);
                assert(-1e-7 <= state_prob && state_prob <= 1 + 1e-7);
                
                state_expect(i) = state_expect(i) + state_prob;
                                
                component_probs = state_component_likelihoods(model, i, outputs(:, t)');
                component_probs = component_probs ./ sum(component_probs);
                state_component_expect(i, :) = state_component_expect(i, :) + state_prob * component_probs;
             
                weighted = bsxfun(@times, repmat(outputs(:, t)', n_components, 1), component_probs');
                state_component_feature_sum(i, :, :) = state_component_feature_sum(i, :, :) +...
                    shiftdim(state_prob * weighted, -1);
                
                weighted_squares = bsxfun(@times, repmat(outputs(:, t)' .^ 2, n_components, 1), component_probs');
                state_component_feature_square_sum(i, :, :) = state_component_feature_square_sum(i, :, :) +...
                    shiftdim(state_prob * weighted_squares, -1);
                                   
                if t + 1 <= size(outputs, 2);
                    for j=1:n_states;
                        transition_expect(i, j) = transition_expect(i, j) +...
                            prefix_probs(t, i) * suffix_probs(t + 1, j) *...
                            model.transitions(i, j) * state_likelihood(model, j, outputs(:, t + 1)');
                    end;
                else
                    last_expect(i) = last_expect(i) + state_prob;
                end;
            end;
        end;
    end;
    
    assert(abs(sum(state_expect) - size([all_outputs{:}], 2)) < 1e-7);
    
    result.transitions = model.transitions;    
    for i=1:n_states;
        for j=i:n_states;
            result.transitions(i, j) =...
                (transition_expect(i, j) + ip.transition_smoother) /...
                (state_expect(i) - last_expect(i) + (n_states - i + 1) * ip.transition_smoother);
        end;
    end;
    
    
    result.mixtures = bsxfun(@rdivide,...
        state_component_expect + ip.mixture_smoother,...
        state_expect + n_components * ip.mixture_smoother);   
    
    result.means = bsxfun(@rdivide, state_component_feature_sum, state_component_expect);
    result.variances = bsxfun(@rdivide, state_component_feature_square_sum, state_component_expect) -...
                            result.means .^ 2 + ones(n_states, n_components, n_features) * ip.variance_smoother;
                        
    check_model(result);
end

function [res] = improve_model_until(model, data, epsilon, n_iters)
    if ~exist('epsilon', 'var')
        n_iters = 1;
        epsilon = 0.1;
    end;
    if ~exist('n_iters', 'var')
        n_iters = 1000;
    end;
    
    cur_quality = n_sequences_log_likelihood(model, data)
    res = model;
    while n_iters > 0;
        res = improve_model(res, data);
        new_quality = n_sequences_log_likelihood(res, data)
        
        if abs(new_quality - cur_quality) / abs(cur_quality) < epsilon;
            break;
        end;
        
        cur_quality = new_quality;
        n_iters = n_iters - 1;
    end;
end

function [] = draw_state(model, state, color)
    n_components = size(model.mixtures, 2);
    for i=1:n_components;
        center = squeeze(model.means(state, i, :))';
        variances = squeeze(model.variances(state, i, :))';
        corner = center - sqrt(variances);
        rectangle('Position', [corner 2 * sqrt(variances)],...
                  'Curvature', [1 1],...
                  'EdgeColor', color,...
                  'LineWidth', 2)
        text(center(1), center(2), mat2str(model.mixtures(state, i), 2));             
    end;
end

function [] = draw_model(model)
    n_states = size(model.transitions, 1);
    colors = [1 0 0; 0 1 0; 0 0 1];
    for i=1:n_states
        draw_state(model, i, colors(i, :));
    end;
end

function [] = draw_sequence(outputs)
    for i=1:size(outputs, 2)
        rectangle('Position', [(outputs(:, i)' - 0.15) 0.3 0.3],...
                  'Curvature', [1 1],...
                  'FaceColor', [0 0 0]);
    end;
end

function [] = draw_component_part(model, state, component, outputs)
    for t=1:size(outputs, 2)
        likes = state_component_likelihoods(model, state, outputs(:, t)');
        probs = likes / sum(likes);
        rectangle('Position', [(outputs(:, t)' - 0.15) 0.3 0.3],...
                  'Curvature', [1 1],...
                  'FaceColor', repmat(probs(component), 1, 3));
    end;
end