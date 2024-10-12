function label = Predict(model, X)
% 使用HNN对X进行预测
    [H_batch, P_batch] = EvaluateClassifier(X', model,2);
    [~,P] = max(P_batch,[],1);
    label=P;
          
 end