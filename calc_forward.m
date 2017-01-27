function [forward, scaling] = calc_forward( transition_prob, hop_prob, init_prob, input_data )
%calc_forward f_k(t)^i = p(x(1:t)^i, s_k(t)^i)の計算を行う
%  f_k(t)^i = p(x_t^i|s_t,k=1) sum_l f_l(t-1)^i a(k,l)によって計算できるのでそれで計算を行う
% transition_prob:遷移確率(K*K)
% hop_prob:各状態におけるホップ確率(1*K)
% init_prob:最初の状態の確率(1*K)
% input_data:入力データ(n*T)
% input_length:データの長さ(n*1)
% forward:全ての時刻における全てのサンプルのfの値(T*K*n)
% scaling:fowardのアンダーフローを防ぐスケーリング係数(n*T)

K = size(transition_prob,1);
[n, T] = size(input_data);

forward = zeros(T*K*n,1);
forward = reshape(forward, T,K,n);

% forward_hat = zeros(T,K,n);
scaling = ones(n,T);

forward(1,:,:) = (ones(n,1) * init_prob)' .* bernoulli_density(input_data(:,1), hop_prob)';
scaling(:,1) = sum(reshape(forward(1,:,:), K,n),1);
forward(1,:,:) = reshape(forward(1,:,:), K,n) ./ (ones(K,1) * scaling(:,1)');

for t = 2:1:T
%     forward(t,:,:) = (transition_prob * reshape(forward(t-1,:,:),K,n) ) .* bernoulli_density(input_data(:,t), hop_prob)';
    tmp = (transition_prob * reshape(forward(t-1,:,:),K,n) ) .* bernoulli_density(input_data(:,t), hop_prob)';
    scaling(:,t) = sum(tmp ,1);
    forward(t,:,:) = tmp ./ (ones(K,1) * scaling(:,t)');
end

end

