classdef HNN < handle
    %双曲流形参考链接https://github.com/mahesobe/BAthesis/blob/cf4ef6a1bb52c2098a972537d5c55e449b4b9066/analyses/Matlab-resources/eeglab2020_0/plugins/clean_rawdata/manopt/manopt/manifolds/hyperbolic/hyperbolicfactory.m
    %------------------------------- Copyright --------------------------------
    % Copyright (c) 2022 BIMK Group. You are free to use the PlatEMO for
    % research purposes. All publications which use this platform or any code
    % in the platform should acknowledge the use of "PlatEMO" and reference "Ye
    % Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
    % for evolutionary multi-objective optimization [educational forum], IEEE
    % Computational Intelligence Magazine, 2017, 12(4): 73-87".
    %--------------------------------------------------------------------------

    properties(SetAccess = private)
        nVisible=0;%输入的解的决策变量维度，每个决策变量作为一个神经元
        nHidden   = 0;%隐藏层节点数目
        Epoch=1000;
        Batchsize=1;
        Momentum  = 0.5;
        LearnRate = 0.001;%一般是1/1000或者1/10000
        labels=[];
        WA    = [];%输入层-》隐藏层
        WB    = [];%隐藏层-》输出层
        gammas={};
        betas={};
        batchsize=1;
    end
    methods
        %% Constructor
        function obj = HNN(nVisible,nHidden,Epoch,Batchsize,LearnRate,Momentum,batchsize)
            obj.nVisible=nVisible;
            obj.nHidden   = nHidden;
            obj.Epoch     = Epoch;
            obj.Batchsize = Batchsize;
            obj.LearnRate = LearnRate;
            obj.batchsize = batchsize;
            obj.Momentum  = Momentum;
            obj.WA = rand(nHidden,nVisible);
            obj.WB = rand(5,nHidden);
            obj.gammas=cell(1,1);
            obj.betas=cell(1,1);
            for i=1:2
                obj.gammas{1}=ones(nHidden,1);
                obj.betas{1}=ones(nHidden,1);
            end
            
        end
        %% Train
        function train(model,xbatch,lbatch,batchnum)
            vW{1} = zeros(size(model.WA));
            vW{2} = zeros(size(model.WB));
            rates=[];
            cnt=1;
            
            for epoch=1:model.Epoch
                for j=1:batchnum
                  %% 前向传播
                    X1=xbatch{j};
                    labels = lbatch{j};
                    m=size(X1,1);
                    
                    [H_batch, P_batch]  = EvaluateClassifier(X1', model,2);
                    [~,P] = max(P_batch,[],1);
                    labs=labels';
                    lambda=0.5;
                    %% 反向传播
                    [grad_W] = ComputeGradients(X1', labs,H_batch, P_batch, model,2, lambda);
                    problemW1.M= poincareballfactory(size(model.WA,1),size(model.WA,2));
                    g_1=(problemW1.M.egrad2rgrad (model.WA,grad_W.W{1}));
                    g_1 = model.LearnRate*g_1;
                    if model.Momentum > 0
                        vW{1} = model.Momentum*vW{1} + g_1;
                        g_1 = vW{1};
                    end
                    model.WA = (problemW1.M.retr(model.WA,-g_1));
                    problemW1.M= poincareballfactory(size(model.WB,1),size(model.WB,2));
                    g_2=(problemW1.M.egrad2rgrad (model.WB,grad_W.W{2}));
                    g_2 = model.LearnRate*g_2;
                    if model.Momentum > 0
                        vW{2} = model.Momentum*vW{2} + g_2;
                        g_2 = vW{2};
                    end
                    model.WB = (problemW1.M.retr(model.WB,-g_2));
                    
                end
                cnt=cnt+1;
            end

            
        end
        
    end
end
