# Ranked-List-Truncation 

We propose a Ranked list truncation framework, including the implementation of multiple existing deep models and the multi-task model modified on these models. We also propose MMOECut for this problem, which achieves the SOTA results, 3.58% higher than the existing SOTA (Attncut).

Since there is still no open source work in this field, we start from the data preprocessing stage and make the whole process code open-source, hoping to avoid researchers from recreating wheels.

## The models we have reproduced include:

* BiCut：An Assumption-Free Approach to the Dynamic Truncation of Ranked Lists（2019-ICTIR）
* Choopy：Cut Transformer For Ranked List Truncation（2020-SIGIR）
* AttnCut：Learning to Truncate Ranked Lists for Information Retrieval（2021-AAAI）

## Multi-task model modified according to the above model:

* MtChoopy
* MtAttnCut

The multi-task model we proposed:
* MMOECut

## Dataset Preprocessing

In this project, we provide a complete processing flow for the robust04 dataset, including word segmentation, cleaning, statistical feature extraction, etc., and complete the production of the dataset for the truncated task in data_prep/data_review.ipynb.

We also complete the training of the retrieval model DRMM-TKS and doc2vec model in another project, which will be gradually open-sourced in the future.

Aiming at the DCG-invalidity of the dataset used by BiCut, Choopy, AttnCut, and other models, we optimized the upstream recall process and used the DRMM-TKS model to obtain a ranked list that is more in line with the modern retrieval model. This dataset has been used in this project.

## Dataset

In dataset/, we show the three datasets used by the framework: the ranked list obtained by using three IR schemes on the robust04 dataset, namely BM25, DRMM, and DRMM-TKS.

Among them, the features used by AttnCut and BiCut are listed separately in the folder, and the .pkl files in the root directory of the dataset/ only contain the relevance scores obtained by the retrieval model.

## Implementation of some deep models

We first reproduced the three existing models of BiCut, Choopy, and Attncut according to the settings of those papers.

In response to the truncation mechanism hypothesis we proposed, we decided to use multi-task learning to achieve this task. First of all, we transformed Choopy and Attncut into a multi-task model according to the shared bottom architecture. Because this architecture avoids the damage to the original model structure to the greatest extent, it only has more branches for auxiliary tasks so that the results can be compared more fairly. 

Finally, we adopted the MMOE architecture proposed by Google and designed the MMOECut model to achieve the SOTA results of this problem.

## Free to add your models、losses and so on

In order to make it easier for researchers to design their own models, loss functions, and metrics, we decouple to the greatest extent:

* One can easily add new models in Models/;
* Add new loss in utils/losses.py;
* Add new metrics in utils/metrics.py.



# 排序列表截断

我们提出一个排序列表截断框架，包括目前已有的多个深度模型的复现以及据此改造的多任务模型，同时，我们也提出了针对该问题的MMOE截断模型，在该问题上达到了SOTA效果，比现有SOTA（Attncut）提升了3.58%。

由于该领域目前仍无开源工作，所以我们从数据预处理阶段开始，将全流程代码进行开源，希望避免研究人员重复造轮子。

我们复现的模型包括：

* BiCut：An Assumption-Free Approach to the Dynamic Truncation of Ranked Lists（2019-ICTIR）
* Choopy：Cut Transformer For Ranked List Truncation（2020-SIGIR）
* AttnCut：Learning to Truncate Ranked Lists for Information Retrieval（2021-AAAI）

根据上述模型改造的多任务模型：

* MtChoopy
* MtAttnCut

我们提出的多任务模型：

* MMOECut

## 数据预处理

我们在该项目中提供了对于robust原始数据集的完整处理流程，包括分词、清洗、统计特征提取等等，并在data_prep/data_review.ipynb中完成截断任务数据集的制作。

同时，我们还在另一个项目中完成了检索模型DRMM-TKS和doc2vec模型的训练，日后将逐步开源。

针对BiCut、Choopy、AttnCut等模型所用数据集的DCG无效性，我们优化了上游的召回过程，并使用DRMM-TKS模型得到了更符合现代检索模型的排序列表，该数据集已经在本项目中得到了使用。

## 数据集

在该文件夹中，我们展示了该框架使用的三个数据集，也就是在robust04数据集上使用三种检索方案得到的排序列表，分别是BM25、DRMM、DRMM-TKS。

其中，AttnCut和BiCut所用的特征在文件夹中单独列出，dataset文件夹根目录的.pkl文件均只包含检索模型得到的相关性分数。

## 模型复现

我们首先根据论文的设定，复现了BiCut、Choopy和Attncut三个现有模型。

针对我们提出的截断机制假设，我们决定采用多任务学习来实现该任务。首先，我们将Choopy和Attncut按照shared bottom架构改造成多任务模型，由于这种架构最大程度上避免了对原本模型结构的破坏，仅仅是多了辅助任务的分支，所以可以更加公平的进行结果对比。

最后，我们采用Google提出的MMOE架构，设计了MMOECut模型，实现了该问题的SOTA效果。

## 添加自定义模型、损失和评估指标

为了方便研究人员设计自己的模型、损失函数和评估指标，我们最大程度上进行了解耦：

* 可以很方便地在Models中添加新的模型；

* 在utils/losses.py中添加新的loss；

* 在utils/metrics.py中添加新的评估指标