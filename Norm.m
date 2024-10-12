 function val=Norm(A)
%  Normalize
    [row,~]=size(A);
    val=zeros(row,1);
    for i=1:row
        t_1=A(i,:);
        val(i,1)=norm(t_1,2);
    end
end