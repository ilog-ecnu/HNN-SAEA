function val=expMap(u,c)
% Exponential map Eqn(5)
   sqrt_c=c.^0.5;
   y_norm =Norm(u);
   y_norm(find(y_norm<1e-5))=1e-5;
   val=tanh(sqrt_c.*y_norm).*u./(sqrt_c.*y_norm);
 end
    