function [est_phi, est_gamma, est_alpha, est_beta, hidden_s1, hidden_s1_s2, energy] = hmm_bernoulli_vb( seed, x_data, gamma, alpha, beta, phi, K )
%hmm_tasep_vb hmm_tasep���f����vb�Ő���
%   seed:�w�K���n�߂鏉���l�̗���
%   x_data:�e�����Ői�񂾂��ǂ������������f�[�^(n*T)
%   gamma:�����m���̎��O���z�̃n�C�p�[�p�����[�^
%   alpha:�����m���̎��O���z�̃n�C�p�[�p�����[�^
%   beta:���s�m���̃n�C�p��
%   phi:�J�ڊm���̃n�C�p��
%   K:��Ԑ�

%%�w�K�ɕK�v�ȕϐ����`
[n,T] = size(x_data);
%%�����̏�����
rng(seed);

%%�w�K�̏����l���߂�
hidden_s1 = zeros(n,T,K);
hidden_s1_s2 = zeros(n,T,K,K);
for i = 1:1:n
    for t = 1:1:T
        if t == 1
            hidden_s1(i,t,:) = dirrnd(ones(1,K));
        else
            hidden_s1_s2(i,t,:,:) = vec2mat(dirrnd(ones(1,K*K)),K);
            hidden_s1(i,t,:) = sum(reshape(hidden_s1_s2(i,t,:,:),K,K),2); %%hidden_s1_s2���k�񂷂��hidden_s1�ɂȂ邽��
        end
    end
end

% disp(sum(sum(hidden_s1_s2(1,2,:,:))));

%%�w�K�J�n
ITERATION = 1000;
% est_phi = zeros(K,K);
% est_gamma = zeros(1,K);
est_alpha = zeros(1,K);
est_beta = zeros(1,K);

for iteration = 1:1:ITERATION
    %%�p�����[�^��
    est_phi = squeeze(sum(sum(hidden_s1_s2,1),2)) + phi;
    est_gamma = squeeze(sum(hidden_s1(:,1,:),1))' + gamma;
    for k = 1:1:K
        est_alpha(1,k) = sum(sum(squeeze(hidden_s1(:,:,k)) .* (x_data==1))) + alpha;
        est_beta(1,k) = sum(sum(squeeze(hidden_s1(:,:,k)) .* (x_data==0))) + beta;
    end
        
    %%�B��ϐ���
    %%���肳�ꂽ�n�C�p�[�p�����[�^��EM�@�Ɠ����`���ɕϊ�����
    quasi_transition_prob = exp(psi(est_phi)-psi(ones(K,1) * sum(est_phi,1)));
    quasi_init_prob = exp(psi(est_gamma)-psi(sum(est_gamma)));
    quasi_hop_prob = exp(psi(est_alpha)-psi(est_alpha+est_beta));
    [forward, scaling] = calc_forward(quasi_transition_prob, quasi_hop_prob, quasi_init_prob, x_data);
    backward = calc_backward(quasi_transition_prob, quasi_hop_prob, quasi_init_prob, x_data, scaling);
    for k = 1:1:K
        hidden_s1(:,:,k) = (reshape(forward(:,k,:), T,n) .* reshape(backward(:,k,:), T,n))';
        for l = 1:1:K
            n_t_density = (quasi_hop_prob(k).^(x_data(:,2:T)==1)) .* ((1-quasi_hop_prob(k)).^(x_data(:,2:T)==0));
            hidden_s1_s2(:,2:T,k,l) = n_t_density .* reshape(forward(1:(T-1),l,:), T-1,n)' .* reshape(backward(2:T,k,:), T-1,n)' .* quasi_transition_prob(k,l) ./ scaling(:,2:T);
        end
    end
    
    %%�G�l���M�[�v�Z
    energy = 0;
    energy = energy + (est_gamma-gamma) * (psi(est_gamma)-psi(sum(est_gamma)))' ...
        + gammaln(sum(est_gamma)) - sum(gammaln(est_gamma));
    energy = energy + (est_alpha-alpha) * (psi(est_alpha)-psi(est_alpha+est_beta))' ...
        + (est_beta-beta) * (psi(est_beta)-psi(est_alpha+est_beta))' ...
        + sum(gammaln(est_alpha + est_beta) - gammaln(est_alpha) - gammaln(est_beta));
    energy = energy + sum(sum( (est_phi - phi) .* (psi(est_phi)-psi(ones(K,1) * sum(est_phi,1))) )) ...
        + sum(gammaln(sum(est_phi,1))) - sum(sum(gammaln(est_phi)));
    energy = energy + K*K*gammaln(phi)-K*gammaln(K*phi)+K*(gammaln(alpha)+gammaln(beta)-gammaln(alpha+beta)) ...
        + K*gammaln(gamma)-gammaln(K*gamma);
    energy = energy -sum(sum(log(scaling)));
%     disp(energy);
end

disp('energy:');
disp(energy);
end

