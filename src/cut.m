function [ digits ] = cut(sig)
    cs = constants();
    extra = mod(length(sig), cs.window_size);
    sig = sig(extra + 1:length(sig));
    
    n_frames = length(sig) / cs.window_size;
    energs = energies(sig);
    
    % -1: start state
    % 0: silence state
    % 1: digit state
    state = -1;
    digit_start = -1;
    last_sound = -1;
    silent_frame_count = 0;
    
    digits = cell(1, 11);
    current_digit = 1;

    for i=1:n_frames;
        if energs(i) < cs.silence_threshold;
            silent_frame_count = silent_frame_count + 1;
            
            if silent_frame_count >= cs.min_frames_beetween_digits;
                if state == 1;
                    if current_digit > cs.n_classes;
                        error(sprintf('Too many digits found, position %d', i))
                    end;
                                        
                    % one silent frame to the left
                    start = max(1, (digit_start - 1 - cs.margin) * cs.window_size + 1);
                    % one silent frame to the right
                    finish = min(length(sig), (last_sound + cs.margin) * cs.window_size);
                    segment_length = finish - start + 1;
                    
                    % sprintf('%d %d', (start - 1) / cs.window_size, finish / cs.window_size)
                    
                    if segment_length <= 4;
                        error(sprintf('%d frames is too short for digit', segment_length));
                    end;
                    
                    digits{current_digit} = sig(start:finish);
                    current_digit = current_digit + 1;
                end;
                state = 0;
            end;
        else;
            last_sound = i;
            
            if state == -1;
                error('no silence in the recording beginning found');
                return;
            end;
            
            if state == 0;
                digit_start = i;
            end
            
            if state == 1;
                if silent_frame_count > cs.max_silent_frames_in_digit;
                    error(sprintf('%s %s, %s %d',...
                          'silence inside the utterance',...
                          int2str(current_digit),...
                          'position',...
                          i));
                    return;
                end;
            end;
            
            silent_frame_count = 0;
            state = 1;
        end;
    end;
    if current_digit <= 11
        error('not all digits were found');
    end;
end

