function backward = calc_backward( transition_prob, hop_prob, init_prob, input_data, scaling )
%calc_backward b_k(t)^i = p(x(t+1:T)^i | s_k(t)^i=1)の計算を行う
%  b_k(t)^i = sum_l p(x_(t+1)^i|s_(t+1),l=1)  b_l(t+1)^i a(l,k)によって計算できるのでそれで計算を行う
% transition_prob:遷移確率(K*K)
% hop_prob:各状態におけるホップ確率(1*K)
% init_prob:最初の状態の確率(1*K)
% input_data:入力データ(n*T)
% input_length:各サンプルデータの長さ(n*1)
% backward:全ての時刻における全てのサンプルのfの値(T*K*n)
% scaling:スケーリング係数(n*T)

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

