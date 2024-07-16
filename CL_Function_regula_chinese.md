### Hi, I'am Muyi Bao

Here, I will put some paper about **Continual Learning**, focusing on Function-Regularization. The classification of the methods refers this survey: A Comprehensive Survey of Continual Learning:Theory,Method and Application 

---


  <details> 
   <summary>
   <b style="font-size: larger;">2.1.2.1 Learning without Forgetting(LwF) </b> <!--  2.1.2.1 Learning without Forgetting(LwF)   -->
   </summary>   
    
   The Paper, published in 2017.11.14: [Learning without Forgetting](https://ieeexplore.ieee.org/abstract/document/8107520)

   我大体记得，很多文章提到Continual Learning的比较规范的定义和基本方法的讨论是在这篇文章进行的，所以我会把一些基本的东西都丢进来

   在开篇讨论了四种最基本的方法和两种这篇文章作者提出来的方法，如图Fig.1和Fig.2。此外还讨论了很多东西，讨论的内容挺多的，好多我都不了解

  - (a) 是原来的模型，以CNN为代表，这里注明了model前面的特征提取层记作θs，后面分类的FC记作θo
  - (b) 为fune-tuning，在新的数据集上进行微调，其中多出来的分类头记作θn，保持其他分类头θo冻结，微调θs，正常训练θn，按照Fig.1所示，这种在原来任务上的表现并不好
  - (c) 为feature extraction，θs和θo保持不变，将一个或多个层的输出作为训练θn的新任务的特征 θs and uo are unchanged,and the outputs of one or more layers are used as features for the new task in training θn，感觉就是冻结原本的模型，只训练θn
  - (d) Joint Learning应该为把所有的数据集的图片都放到一起，然后一起进行训练，这并不是CL，而是可以大体作为CL任务的性能上界
  - (e) Less-forgetting Learning，这好像是作者的前面一篇工作，由于没有看，所以也不知道说的啥
  - (f) 就是这篇工作，简单来说，是第一篇使用knowledge distinction的方法在Continual Learning身上 

<img src="https://github.com/BaoBao0926/Paper_reading/blob/main/Image/2.Continual_Learning/Regularization-Based_Approach/Funtion/LwF1.png" alt="Model" style="width: 1000px; height: auto;"/>

整体的方法实际上非常简单，也就是微调+知识蒸馏knowledge distinction的结合

  - 微调，从Fig.2f可以看到，θo和θs都是进行微调的，并不是冻结的，只有新任务的参数是随机初始化+训练
    - 不过对于具体的训练过程而言，前20个epoch会把除了θn以外的全部冻结，只训练θn，这是进行warm-up stage
  - 知识蒸馏，简单来说就是在实际预测造成的硬损失之外，让上一个模型也跑一次结果，用这次的预测值与上一次模型输出的预测值进行一个损失
    - 按照知识蒸馏的原文，会进行一个 软化？的操作，如Function(4)
  - 根据伪代码Fig.3中的最后一行，可以看到一个R，不过文中只有这一句话描述了这个R we train the network to minimize loss for all tasks and regularization R using stochastic gradient descent.The regularization R corresponds to a simple weight decay of 0.0005。说的不是很清楚，我也不太理解这是啥意思
  - 对于模型本身而言没有太多要求，直接使用的是AlexNet进行的，不过后面也做了一个关于VGG的实验


<img src="https://github.com/BaoBao0926/Paper_reading/blob/main/Image/2.Continual_Learning/Regularization-Based_Approach/Funtion/LwF2.png" alt="Model" style="width: 1000px; height: auto;"/>

数据集
- ImageNet， VOC 2012 Image Classification(VOC), Caltech-UCSD-Birds-200-2011 fine-grained classification(CUB), MIT indoor scene classification(Scene) , MNIST
- 多数据集CL学习： ImageNet->VOC/CUB/Scenes/MNIST， Place365/VOC/CUB/Scenes/MNIST

整体来说，LwF这个方法是最好的，除此之外，还讨论了一些东西
  - 在这一坨数据集中，如果数据集之间越不一样，也就是domian越不一样，那么整体的效果就会越差，用Joint Training的差距进行比较，这点主要体现在ImageNet->MNIST上
  - 数据集的大小：这篇文章实验了数据集的大小是否影响影响，使用的是CUB添加到ImageNet里面，训练网络时，使用的是30%， 10% 3%的CUB大小。结果都由于fine-tune，方法之间的差异随着使用的数据增多而增加，尽管相关性不确定，这个可以从Fig.5中看到，当数据越多的时候，越离散
  - 第三个讨论的东西是，对于最后分类头的MLP的讨论，可以从Fig.6看到
    - Fig.6a中是 choice of task-specific layer，也就是整体的特征提取层结束之后，每多一个分类任务就多一个单独的MLP出来用于分类
    - Fig.6b是network expansion，他的做法是让前几层MLP是连接在一起的，多一个分类种类，就让这些共享参数的MLP多一些node出来(具体是1024个node)，使用Net2Net的方法进行初始化复制出来的新node
    - 这个network expandsion在Growing a Brain的工作中，通过在FC7的4096个node中添加1024个新node，可以增加0.53%，添加2048个新node，可以增加0.88%的性能，参数量增加21%。不过LwF添加2048个新node指挥增加2.7%的参数量，总的来说，这个network expansion是有用的
- 这篇文章在conclusion中讨论了自己的5个limitations
  - 他不能正确处理在一个domain that continually changing on a spectrum，必须要对任务进行枚举emumerated，这个类似于multitaks learning，并且每一个样本都需要有标签。我理解的是，如果都是预测狗，但是两个数据集一个是真实世界狗，一个是卡通狗，那么就不能很好的处理
  - 不是以流stream的形式进行的，需要数据集在train之前就收集好
  - 学习新任务的能力是有限的，旧任务的性能会逐渐下降
  - 在VGG上训练时，与联合学习的差距会变得较大
  - LwF的性能很大程度上取决于新任务数据与旧任务数据之间的相似程度，对于VOC来说，与MIT indoor scenes有点相似，与CUB不相似(只有鸟类图片)，与MNIST(没有相似之处)，也就可以从数据中看到差距非常大

<img src="https://github.com/BaoBao0926/Paper_reading/blob/main/Image/2.Continual_Learning/Regularization-Based_Approach/Funtion/LwF3.png" alt="Model" style="width: 1000px; height: auto;"/>
  
</details>




