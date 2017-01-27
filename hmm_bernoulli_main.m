function hmm_bernoulli_main( transition_prob, init_prob, hop_prob, n, time_length, data_seed, learning_seed, gamma, alpha, beta, phi, K )
%hmm_tasep_main hmm_tasep‚ÌƒƒCƒ“ŠÖ”
%   Ú×à–¾‚ğ‚±‚±‚É‹Lq

[x_data, label_data] = generate_data_hmm_bernoulli(transition_prob, init_prob, hop_prob, n, time_length, data_seed);

% disp(label_data);
% disp(size(x_data));
dlmwrite('all_data.csv', x_data);
dlmwrite('all_data.csv', label_data, '-append', 'roffset', 1);
% 
SEED_NUM = 10;
min_energy = realmax;
for seed_num = 1:1:SEED_NUM
    [est_phi, est_gamma, est_alpha, est_beta, hidden_s1, ~, energy] = hmm_bernoulli_vb(learning_seed+seed_num-1, x_data, gamma, alpha, beta, phi, K);
    if energy < min_energy
        min_energy = energy;
        min_phi = est_phi;
        min_gamma = est_gamma;
        min_alpha = est_alpha;
        min_beta = est_beta;
        min_hidden_s1 = hidden_s1;
    end
end


mean_transition_prob = min_phi ./ (ones(K,1) * sum(min_phi,1));
mean_hop_prob = min_alpha ./ (min_alpha + min_beta);
mean_gamma = min_gamma / sum(min_gamma);

disp('transition prob:');
disp(mean_transition_prob);
disp('hop prob:');
disp(mean_hop_prob);
disp('initial prob:');
disp(mean_gamma);

[~,max_ind] = max(min_hidden_s1,[],3);
dlmwrite('est_label.csv', max_ind);
% 
% disp(squeeze(hidden_s1(1,:,:)));

% func_plot_cell_automaton(cell_automaton);

end

