classdef HNNMCEAD < ALGORITHM
    % <multi/many> <real> <expensive>
    % Multiple classifiers-assisted evolutionary algorithm based on decomposition
    % delta  --- 0.9 --- The probability of choosing parents locally
    % nr     ---   2 --- Maximum number of solutions replaced by each offspring
    % Rmax   ---  10 --- Maximum repeat time of offspring generation
    
    %------------------------------- Reference --------------------------------
    % T. Sonoda and M. Nakata, Multiple classifiers-assisted evolutionary
    % algorithm based on decomposition for high-dimensional multi-objective
    % problems, IEEE Transactions on Evolutionary Computation, 2022.
    %--------------------------------------------------------------------------
    
    % This function is written by Masaya Nakata
    
    methods
        function main(Algorithm, Problem)
            %% Parameter setting
            [delta, nr, R_max] = Algorithm.ParameterSet(0.9, 2, 10);
            
            %% Generate the weight vectors
            [W, Problem.N] = UniformPoint(Problem.N, Problem.M);
            
            %% Detect the neighbours of each solution
            T      = ceil(Problem.N / 10);
            %计算两两的欧氏距离
            B      = pdist2(W, W);
            
            [~, B] = sort(B, 2);
            B      = B(:, 1 : T);%欧式距离排序从小到大的种群下标
            
            %% Initialize population
            PopDec     = lhsamp(Problem.N, Problem.D); % Latin Hypercube Sampling in our experimental environment (PlatEMO ver2)
            Population =  Problem.Evaluation(repmat(Problem.upper - Problem.lower, Problem.N, 1) .* PopDec + repmat(Problem.lower, Problem.N, 1));
            A          = Population;
            Z          = min(Population.objs, [], 1);
            
            %% Define SVM
            svm_list = SVM(Problem);
            epsilon=0.05;
            cnt=1;
            archive=cell(3,1);
            N=Problem.N;
            if N>90
                batchsize=25; %N=91
            else
                batchsize=25; %N=77 A setting of 20 is recommended for WFG, and 25 is recommended for DTLZ
            end
            %% HNN model construction
            model=HNN(Problem.D,20,50,size(A,2),0.05,0.5,batchsize);
            %% Optimization
            while Algorithm.NotTerminated(A)
                %% 建模成分类任务
                lbatch=labelbatch(A,epsilon,batchsize);
                [xbatch,batchnum]=dataBatch(A.decs,batchsize);
                model.train(xbatch,lbatch,batchnum);
                
                %% data augument 干扰 lables 不变
                for i=1:13
                    Decs  = repmat(A.decs,1,1);
                    ind=randperm(size(Decs,2),1);
                    Decs(:,ind) = unifrnd(Problem.lower(ind),Problem.upper(ind),size(Decs,1),1);
                    Decs_1=Decs-A.decs;
                    [xbatch,batchnum]=dataBatch(Decs,batchsize);
                    model.train(xbatch,lbatch,batchnum);
                end
                for j=1:3
                    t_a=archive{j};
                    if size(t_a,1)~=0
                        lbatch=labelbatch(t_a,epsilon,batchsize);
                        [xbatch,batchnum]=dataBatch(t_a.decs,batchsize);
                        model.train(xbatch,lbatch,batchnum);
                    end
                end
                index=mod(cnt,3);
                archive{index+1}=A;
                cnt=cnt+1;
   
                for i = 1 : Problem.N
                
                    %% Model-construction 
                    svm_list(i) = svm_list(i).ModelConstruction(A, B(i, :), W, Z);
                    
                    %% Choose the parents
                    if rand < delta
                        P = B(i, randperm(end));%选择T个随机打乱的种群索引矩阵
                    else
                        P = randperm(Problem.N);
                    end
                    
                    %% Solution-generation
                    [cans, y_i,score]  = SolutionGeneration(Problem,Population, P, model,svm_list(i), R_max, i);
                    
                    %% Evaluate offspring
                    y_i = Problem.Evaluation(y_i);
                    
                    %% Update the reference point
                    Z = min(Z, y_i.obj);
                    
                    %% Update population and archive
                    g_old = max(abs(Population(P).objs - repmat(Z, length(P), 1)) .* W(P, :), [], 2);
                    g_new = max(repmat(abs(y_i.obj - Z), length(P), 1) .* W(P, :), [], 2);
                    Population(P(find(g_old >= g_new, nr))) = y_i;
                    
                    A = [A, y_i];
                    
                    %% Check termination criteria
                    Algorithm.NotTerminated(A);
                end
                
            end
        end
    end
end