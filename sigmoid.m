function s = sigmoid(v,nVar)
    s = zeros(1,nVar);
    for l=1:nVar
        s(:,l) =1/(1+exp(-v(:,l)));
    end
end