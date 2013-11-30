hmm = hmmfuncs();
ml = mlfuncs();
global g_n_states;
g_n_states = 3;
constants()

[utters3, features3, labels3] = loaddir('sounds3');

accs = [];
for t=1:10;
    cms = ml.cross_validate(features3, labels3, 4, [t t]);
    rp_train = ml.confmat2report(cms.confmat_train);
    rp_test = ml.confmat2report(cms.confmat_test);
    
    accs(t) = rp_test.avacc;
    sprintf('ACC: %f', rp_test.avacc)
    sprintf('OVERFIT: %f', rp_train.avacc)
    
    [~, ~, ci, ~] = normfit(accs(1:t));
    sprintf('CI: %f %f', ci)
end;
