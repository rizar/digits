hmm = hmmfuncs();
ml = mlfuncs();
global g_n_states;
g_n_states = 3;
constants()

[utters3, features3, labels3] = loaddir('sounds3');

accs = [];
for t=1:10;
    rp = ml.confmat2report(ml.cross_validate(features3, labels3, 4, [t t]));
    accs(t) = rp.avacc;
    sprintf('ACC: %f', rp.avacc)
    
    [~, ~, ci, ~] = normfit(accs(1:t));
    sprintf('CI: %f %f', ci)
end;
