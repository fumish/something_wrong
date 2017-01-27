function [est_phi, est_gamma, est_alpha, est_beta, hidden_s1, hidden_s1_s2, energy] = hmm_bernoulli_vb( seed, x_data, gamma, alpha, beta, phi, K )
%hmm_tasep_vb hmm_tasepモデルをvbで推定
%   seed:学習を始める初期値の乱数
%   x_data:各時刻で進んだかどうかが入ったデータ(n*T)
%   gamma:初期確率の事前分布のハイパーパラメータ
%   alpha:成功確率の事前分布のハイパーパラメータ
%   beta:失敗確率のハイパラ
%   phi:遷移確率のハイパラ
%   K:状態数

%%学習に必要な変数を定義
[n,T] = size(x_data);
%%乱数の初期化
rng(seed);

%%学習の初期値を定める
hidden_s1 = zeros(n,T,K);
hidden_s1_s2 = zeros(n,T,K,K);
for i = 1:1:n
    for t = 1:1:T
        if t == 1
            hidden_s1(i,t,:) = dirrnd(ones(1,K));
        else
            hidden_s1_s2(i,t,:,:) = vec2mat(dirrnd(ones(1,K*K)),K);
            hidden_s1(i,t,:) = sum(reshape(hidden_s1_s2(i,t,:,:),K,K),2); %%hidden_s1_s2を縮約するとhidden_s1になるため
        end
    end
end

% disp(sum(sum(hidden_s1_s2(1,2,:,:))));

%%学習開始
ITERATION = 1000;
% est_phi = zeros(K,K);
% est_gamma = zeros(1,K);
est_alpha = zeros(1,K);
est_beta = zeros(1,K);

for iteration = 1:1:ITERATION
    %%パラメータ側
    est_phi = squeeze(sum(sum(hidden_s1_s2,1),2)) + phi;
    est_gamma = squeeze(sum(hidden_s1(:,1,:),1))' + gamma;
    for k = 1:1:K
        est_alpha(1,k) = sum(sum(squeeze(hidden_s1(:,:,k)) .* (x_data==1))) + alpha;
        est_beta(1,k) = sum(sum(squeeze(hidden_s1(:,:,k)) .* (x_data==0))) + beta;
    end
        
    %%隠れ変数側
    %%推定されたハイパーパラメータをEM法と同じ形式に変換する
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
    
    %%エネルギー計算
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

