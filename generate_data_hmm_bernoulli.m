function [x_data, label_data] = generate_data_hmm_bernoulli( transition_prob, init_prob, hop_prob, n, time_length, seed )
%generate_data_hmm_bernoulli 与えられたパラメータにしたがったベルヌーイ隠れマルコフモデルのデータを生成
% transition_prob:遷移確率(K*K) (次の時刻の状態)*(前の時刻の状態)
% init_prob:初期状態の確率(1*K)
% hop_prob:ホップ確率(1*K)
% n:データの個数
% cell_num:セルの数
% time_length:時間幅

%%データ生成の初期設定
rng(seed);

x_data = zeros(n,time_length);
label_data = zeros(n,time_length);

for t = 1:1:time_length
    %%状態を決定する    
    if t == 1
        y = mnrnd(1,init_prob,n);
        [label_data(:,1),~] = find(y');
    else
        next_state_prob = transition_prob(:,label_data(:,t-1))';
        y = mnrnd(1,next_state_prob);
        [label_data(:,t),~] = find(y');
    end

    %%コインフリップ
    x_data(:,t) = binornd(1,hop_prob(label_data(:,t))');    
end

end

