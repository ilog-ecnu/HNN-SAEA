function lbatch=labelbatch(X,epsilon,batchsize)
%这里的X是population,获取每个population batch中每个个体的对应label
    N=size(X,2);
    batchnum=ceil(N/batchsize);
    lbatch=cell(batchnum,1);
    for i=1:batchnum
        l=(i-1)*batchsize+1;
        r=i*batchsize;
        if r>N
            r=N;
        end
        t_1=X(:,l:r);
        lbatch{i}=preprocess(t_1,epsilon);
    end
end