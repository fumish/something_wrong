function backward = calc_backward( transition_prob, hop_prob, init_prob, input_data, scaling )
%calc_backward b_k(t)^i = p(x(t+1:T)^i | s_k(t)^i=1)�̌v�Z���s��
%  b_k(t)^i = sum_l p(x_(t+1)^i|s_(t+1),l=1)  b_l(t+1)^i a(l,k)�ɂ���Čv�Z�ł���̂ł���Ōv�Z���s��
% transition_prob:�J�ڊm��(K*K)
% hop_prob:�e��Ԃɂ�����z�b�v�m��(1*K)
% init_prob:�ŏ��̏�Ԃ̊m��(1*K)
% input_data:���̓f�[�^(n*T)
% input_length:�e�T���v���f�[�^�̒���(n*1)
% backward:�S�Ă̎����ɂ�����S�ẴT���v����f�̒l(T*K*n)
% scaling:�X�P�[�����O�W��(n*T)

K = size(transition_prob,1);
[n, T] = size(input_data);

backward = zeros(T*K*n,1);
backward = reshape(backward, T,K,n);
for i = 1:1:n
    backward(T,:,i) = ones(K,1);
end

for t = (T-1):-1:1
    backward(t,:,:) = (transition_prob' * ( reshape(backward(t+1,:,:),K,n ) ...
        .* bernoulli_density(input_data(:,t+1), hop_prob)')) ...
        ./ (ones(K,1) * scaling(:,(t+1))');
end
end

