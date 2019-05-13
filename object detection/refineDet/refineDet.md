## refineDet



### motivation

`one stage`目标检测准确率低的原因是因为 `class imbalance problem`。

已有的一些解决方案｛

- object prior constraint[24]
- reshaping the standard cross entropy loss to focus on hard examples[28]
- max-out labeling mechanism[53]

｝

`two stage` 工作的advantages｛

- two stage 结构做 sampling heuristics 采样探索 来处理 class imbalance problem
- 两步级联地去回归 object box parameters
- two stage features 来描述objects--> RPN 特征 预测二分类（object / background）， stage2 预测 多分类（BG and objects classes）  **本文借鉴了这种两次分类的好处**

｝



### structure 两个模块

- anchor refinement module ARM 
- objection detection module  ODM







