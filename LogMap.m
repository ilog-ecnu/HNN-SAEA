 function res= LogMap(  x, c)
 % Logarithmic map Eqn(4)
     sqrt_c=c.^0.5;
     p_norm=Norm(x);
     p_norm(find(p_norm<1e-5))=1e-5;
     scale=1./sqrt_c.*atanh(sqrt_c.*p_norm)./p_norm;
     res=scale.*x;
 end
 function val=arcosh(X)
     X1=X.^2-1;
     X1(find(X1<1e-15))=1e-15;
     val=log(X1.^0.5+X);
 end