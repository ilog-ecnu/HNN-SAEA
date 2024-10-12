function accuracy = ComputeAccuracy(X, y, NetParams, varargin)
    n = size(y, 1);

    acc = zeros(1, n);

    if nargin < 4
        [~, P] = EvaluateClassifier(X, NetParams);
    else
        [~, P] = EvaluateClassifier(X, NetParams, varargin{1}, varargin{2});
    end

    prediction = Argmax(P);
    

    for i = 1:n
        if prediction(i) == y(i)
            acc(i) = 1;
        else
            acc(i) = 0;
        end
    end

    accuracy = sum(acc)/n;
end
function argmax = Argmax(matrix)
    [~, argmax] = max(matrix);
end