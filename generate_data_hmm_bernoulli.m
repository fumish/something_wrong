function [x_data, label_data] = generate_data_hmm_bernoulli( transition_prob, init_prob, hop_prob, n, time_length, seed )
%generate_data_hmm_bernoulli �^����ꂽ�p�����[�^�ɂ����������x���k�[�C�B��}���R�t���f���̃f�[�^�𐶐�
% transition_prob:�J�ڊm��(K*K) (���̎����̏��)*(�O�̎����̏��)
% init_prob:������Ԃ̊m��(1*K)
% hop_prob:�z�b�v�m��(1*K)
% n:�f�[�^�̌�
% cell_num:�Z���̐�
% time_length:���ԕ�

%%�f�[�^�����̏����ݒ�
rng(seed);

x_data = zeros(n,time_length);
label_data = zeros(n,time_length);

for t = 1:1:time_length
    %%��Ԃ����肷��    
    if t == 1
        y = mnrnd(1,init_prob,n);
        [label_data(:,1),~] = find(y');
    else
        next_state_prob = transition_prob(:,label_data(:,t-1))';
        y = mnrnd(1,next_state_prob);
        [label_data(:,t),~] = find(y');
    end

    %%�R�C���t���b�v
    x_data(:,t) = binornd(1,hop_prob(label_data(:,t))');    
end

end

