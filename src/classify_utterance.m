function [digit_recognized] = classify_utterance(model, test_utterance)
    ml = mlfuncs();

    recording = audioread(test_utterance);
    features = featurize(recording);
    prediction = ml.predict(model, {features});
    
    if prediction <= 2;
        digit_recognized = 0;
    else
        digit_recognized = prediction - 2;
    end;
end

