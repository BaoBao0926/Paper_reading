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


<img src="https://github.com/BaoBao0926/Paper_reading/blob/main/Image/2.Continual_Learning/Regularization-Based_Approach/Funtion/LwF1.png" alt="Model" style="width: 600px; height: auto;"/>
    
</details>




