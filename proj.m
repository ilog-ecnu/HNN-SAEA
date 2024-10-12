function val=proj(x,c) 
% Identity map since the embedding space is the tangent space
    norm=Norm(x);
    norm(find(norm<1e-5))=1e-5;
    eps=10e-8;
    maxnorm=(1-eps)/(c.^0.5);
    cond=norm>maxnorm;
    projected=x./norm.*maxnorm;
    x(cond)=projected(cond);
    val=x;
end