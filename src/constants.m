function [ values ] = constants()
    values = struct('n_features', 14,...
                    'n_classes', 11,...
                    'window_size', 256,...
                    'window_step', 128,...
                    'margin', 3,...
                    'silence_threshold', 0.05,...
                    'min_frames_beetween_digits', 10,...
                    'max_silent_frames_in_digit', 5,...
                    'n_states', 3,...
                    'n_components', 3,...
                    'hmm_epsilon', 1e-2,...
                    'do_normalize_features', 1);
                
    global n_states;
    if ~isempty(n_states);
        values.n_states = n_states;
    end;
    
    global n_components;
    if ~isempty(n_components);
        values.n_components = n_components;
    end;
    
    global hmm_epsilon;
    if ~isempty(hmm_epsilon);
        values.hmm_epsilon = hmm_epsilon;
    end;
end

