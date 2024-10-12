function [Xs, P] = EvaluateClassifier(X, model, k)
    W{1}=model.WA;
    W{2}=model.WB;
    gammas = model.gammas;
    betas = model.betas;

    n = size(X, 2);
    X=LogMap(X,1);
    %% un-norm scores of layer 1
    mu = -1;
    v = -1;

    Xs{1} = max(0, W{1} * X);

    for i=2:(k-1)
        Xs{end + 1} = max(0, W{i} * Xs{i-1} );
    end

    S{1} = W{1} * X ;
    S_tilde=S{1};
    S_tilde_1=expMap(S_tilde,1);
    S_tilde_1=proj(S_tilde_1,1);
    % ReLu
    Xs{1} = max(0, S_tilde_1);
    % Final linear transformation
    t_1=Xs{k-1};
    t_1=LogMap(t_1,1);
    s = W{k} * t_1 ;
    s_1=expMap(s,1);
    s_1=proj(s_1,1);
    s_1=LogMap(s_1,1);
    P_1 = softmax(s_1);
    P_1=expMap(P_1,1);
    P=proj(P_1,1);
end
