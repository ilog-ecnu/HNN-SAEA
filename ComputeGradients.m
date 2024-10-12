function Grads = ComputeGradients(X_batch, labels, Xs_batch, P_batch,model,k, lambda)

    n_batch = size(X_batch, 2);

    W{1}=model.WA;
    W{2}=model.WB;

    grad_W = cell(1, k);


    % Propagate the gradient through the loss and softmax operations
    G_batch = - (labels - P_batch);
    for l=k:-1:2
        % Compute grad of J wrt W{l}
        grad_W{l} = (G_batch * Xs_batch{l-1}')/n_batch + 2 * lambda * W{l};
      

        % Propagate G_batch to the previous layer
        G_batch = W{l}' * G_batch;
        G_batch( Xs_batch{l-1} <= 0 ) = 0;
    end

    grad_W{1} = (G_batch * X_batch')/n_batch + 2 * lambda * W{1};

    Grads.W = grad_W;

end