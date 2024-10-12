function S_hat = BatchNormalize(S, mu, v)
    S_hat = diag( ( v + eps ).^(-1/2) ) * (S - mu);
end