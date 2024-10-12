

## 论文

Li B, Yang Y, Hong W, et al. Hyperbolic Neural Network Based Preselection for Expensive Multi-Objective Optimization[J]. IEEE Transactions on Evolutionary Computation, 2024.

## 算法框架

### A:流程图

#### A.1: 主流程

![fig-HNN-p1](https://github.com/user-attachments/assets/5ba735f8-4cc0-4499-a499-fca318d71b84)

#### A.2: HNN based Preselection 细节

HNN-SAEAs通过基于HNN（Hopfield神经网络）的预选机制进行后代生成，包括GA（遗传算法）操作和基于HNN的预选操作。

![fig-HNN-p2](https://github.com/user-attachments/assets/fb42c13e-3d20-42b6-a2c7-2faa02a5a381)


### B: 算法主流程伪代码

<img width="244" alt="HNNflowchart" src="https://github.com/user-attachments/assets/60f609bd-049e-4d01-8c82-9b30df00b99e">


## 运行

**代码主函数：**HNNMCEAD.m

**脚本:**

```matlab
addpath('.');
LSMOP = {@LSMOP3,@LSMOP4,@LSMOP5,@LSMOP6,@LSMOP7,@LSMOP8,@LSMOP9,@LSMOP1,@LSMOP2};
Problem={LSMOP};
Alg = {@HNNMCEAD};
M=3;
D = [100];
N = [100];
len=length(D);
maxFE=300;
save=10;
run_count=5;
for alg_index=1:length(Alg)
    for pro_index=1:length(Problem)
        for pro_inner_index=1:length(Problem{pro_index})
            for r=1:run_count
                for i=1:len
                    platemo('algorithm',Alg{alg_index},'problem',Problem{pro_index}{pro_inner_index},'M',M,'D',D(i),'N',N(i),'maxFE',maxFE(i),'save',save);
                end
            end
        end
    end
end
```



## 版本

**PlatEMO version:** PlatEMO 4.2

**Matlab version:**  Matlab R2022b

