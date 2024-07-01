
## Hi, I am [Muyi Bao](https://github.com/BaoBao0926/BaoBao0926.github.io)

[English](https://github.com/BaoBao0926/Paper_reading) | [简体中文](https://github.com/BaoBao0926/Paper_reading/blob/main/Readme_Chinese)

---

我会在这里写下来我读过的文章的创新点。我复现过的文章的代码在这个[仓库](https://github.com/BaoBao0926/Overview-of-Reproduced-Project)


---

<details>      <!--    -----------------------------------------  1.Mamba   -------------------------------------------------------  -->
    <summary>
   <b> 1.Mamba </b> 
   </summary>   
   
   <br />


<details>      <!--    -----------------------------------------  1.1 Vision Mamba   -------------------------------------------------------  -->
    <summary>
   <b> 1.1 Vision Mamba </b> 
   </summary>   
   
   <br />


<details> 
   <summary>
   <b> 1.1.1 Vision Mamba Backbone Network</b>  <!--    -----------------------------------------  1.1.1 Vision Mamba Backbone Network  -------------------------------------------------------  -->
   </summary>   
   
   <br />


  <details> 
   <summary>
   <b style="font-size: larger;">1.1.1.1 Vision Mamba(Vim) </b> <!--   1.1.1.1  Vision Mamba(Vim)   -->
   </summary>   
    
   The Paper: [Vision Mamba: Efficient Visual Representation Learning with Bidirectional State Space Model](https://arxiv.org/abs/2401.09417)

  这篇文章与Vision Transformer非常相似

贡献：

- Mamba block里面的扫描方向是双向的，前向和反向，反向通过使用flip()实现
- 使用了0，1或者2两个cls token：
  - 0个cls token，使用max mean进行最后预测
  - 1个cls token，可以把cls token插入在头部，尾部，中间，或者随机位置
  - 2个cls token，一个尾部，一个头部
   

<img src="https://github.com/BaoBao0926/Overview-of-Reproduced-Project/blob/main/Code/009.Vision%20Mamba(Vim)/architecture.png" alt="Model" style="width: 600px; height: auto;"/>
    
</details>


  <details> 
   <summary>
   <b style="font-size: larger;">1.1.1.2 Visual Mamba(VMamba) </b>    <!--   1.1.1.2  Visual Mamba(VMamba)   -->
   </summary>   
    
   The Paper: [VMamba: Visual State Space Model](https://arxiv.org/abs/2401.10166)

整体架构与Swim Transoformer相似

贡献：

   - 2D Selective Scan(SS2D): 使用了四个方向，竖着的在代码中是通过交换宽和高来实现的
   - VSS Block: 改变了Mamba block内部的运行方式
   - Architecutre: 使用了类似于Swim Transformer的架构

<img src="https://github.com/BaoBao0926/Paper_reading/blob/main/Image/1.Mamba/1.1%20VisionMamba/1.1.1%20Backbone_network/VMamba.png" alt="Model" style="width: 600px; height: auto;"/>
    

</details>


<details> 
   <summary>
   <b style="font-size: larger;">1.1.1.3 Mamba-ND </b>          <!--   1.1.1.3  Mamba-ND   -->
   </summary>   
    
   The Paper: [VMaMamba-ND: Selective State Space Modeling for Multi-Dimensional Data](https://arxiv.org/abs/2402.05892)

 
  这篇文章认为自己是多维数据的general-implementation
   
   贡献:

   - 提出了Later-level和Block-level两个level去进行scan：
     - Layer-level: 在一个mamba block里面，有多个不同方向的通路
     - Block-level: 现在假设一个mamba block只有一个方向，现在有多个不同方向的block按照不同的顺序组和到了一起形成了整体架构
   - In Block-level, 尝试了 H+H-W+W-T+T-, [H+H-][W+W-][T+T-], [H+H-W+W-][T+T-]. 发现 H+H-W+W-T+T- 是最好的. 不过这部分的代码很难理解
   - 使用了三种scan-dactorization策略。但是我没有理解是干嘛的

<img src="https://github.com/BaoBao0926/Paper_reading/blob/main/Image/1.Mamba/1.1%20VisionMamba/1.1.1%20Backbone_network/Mamba-ND.png" alt="Model" style="width: 600px; height: auto;"/>
    

</details>

<details> 
   <summary>
   <b style="font-size: larger;">1.1.1.4 Local Mamba </b>          <!--   1.1.1.4  Local Mamba   -->
   </summary>   
    
   The Paper: [LocalMamba: Visual State Space Model with Windowed Selective Scan](https://arxiv.org/pdf/2403.09338)

  基于Vim和VMamba为baseline，有更好的表现
   
   贡献:

   - 提出了Fig.1c的方式去进行scan，一个一个local window进行扫描，有两个大小，2 * 2 和 7 * 7
   - 在一个mamba block里面，使用了四个分路，方向和怎么选择如下
     - 方向有：vertical, vertical-filp, horizontial, horizontial-flip, 2*2 window, 7*7 window
     - 如何选择：inspired by DARTS, 使用了一个可训练的网络和可学习的factor，来选择一个block里面应该选择上面8个方向的那四个
   - SCAAttm, spatial and channel attention modules: this module can enhance the itegration of diverse features and eliminate extraneous information shown in Fig.3.b. 这似乎是一种注意力机制模块

<img src="https://github.com/BaoBao0926/Paper_reading/blob/main/Image/1.Mamba/1.1%20VisionMamba/1.1.1%20Backbone_network/LocalMamba.png" alt="Model" style="width: 600px; height: auto;"/>
    
   <br />

</details>


</details>      <!--    -----------------------------------------  1.1.1 Vision Mamba Backbone Network  -------------------------------------------------------  -->


<details> 
   <summary>
   <b style="font-size: larger;">1.1.2 Segementation in Medical Image </b>  <!--    -----------------------------------------  1.1.2  Segementation in Medical Image  ------------------------------------------  -->
   </summary>   
    
<details> 
   <summary>
   <b style="font-size: larger;">1.1.2.1 U-Mamba </b>          <!--   1.1.2.1  U-Mamba   -->
   </summary>   
    
   The Paper: [U-Mamba: Enhancing Long-range Dependency for Biomedical Image Segmentation](https://arxiv.org/pdf/2401.04722)

贡献：

- 整体架构使用的是U-Net的架构，应该是作为第一篇基于Mamba的U-net的分割模型，手快就是好
- 使用了nnUnet的架构，可以自动适应数据集
- Mamba block稍微改动了一下，如图Fig.1里面的样子
    - x = x + LeakyRelu(Conv(x))    [B,C,H,W,D] 
    - x = LayerNorm(Flatten(x))     [B,L,C], L = C * H * W
    - x = SiLU(linear(x)) * SSM(SiLU(1D Conv(Linear(x))))    [B,L,C]
    - x = Linear(x)    [B,L,C]
    - x = Reshape(x)    [B,C,H,W,D]
  

<img src="https://github.com/BaoBao0926/Paper_reading/blob/main/Image/1.Mamba/1.1%20VisionMamba/1.1.2%20Segmentation%20in%20medical%20image/U-Mamba.png" alt="Model" style="width: 600px; height: auto;"/>

使用的数据集：

    - MICCAI 2022 FLARE Challenge
    
    - MICCAI 2022 AMOS Challenge
    
    - MICCAI 2017 EndoVis Challenge
    
    - NuerIPS 2022 Cell Segmentation Challenge

   <br />

</details>

<details> 
   <summary>
   <b style="font-size: larger;">1.1.2.2 SegMamba </b>          <!--   1.1.2.2 SegMamba   -->
   </summary>   
    
   The Paper: [SegMamba: Long-range Sequential Modeling Mamba For 3D Medical Image Segmentation](https://arxiv.org/pdf/2401.13560)

贡献：

- 整体架构使用的是U-Net的架构
- Mamba block改成了TSMamba Block，如图Fig.2里面的样子，里面涉及了一些模块
    - input x is [C,D,H,W]
    - x = GSC(x) = x + Conv3d_333(Conv3d_333(x) * Conv3d_111(X)), 每一个卷积都代表着 Norm->Conv3D->Nonlinear
    - x = x + LayerNorm(ToM(x))
        - ToM(x)为Mamba模块，其中有三个方向，如Fig.3b所示，forward，reverse和inter-wise，这个inter-wise代表的是竖着的
    - x = x + MLP(LayerNorm(x))
  

<img src="https://github.com/BaoBao0926/Paper_reading/blob/main/Image/1.Mamba/1.1%20VisionMamba/1.1.2%20Segmentation%20in%20medical%20image/SegMamba.png" alt="Model" style="width: 600px; height: auto;"/>

使用的数据集：

    - CRC-500: 文章提的
    
    - BraTS2023 Dataset

    - AIIB2023 Dataset

   <br />

</details>


<details> 
   <summary>
   <b style="font-size: larger;">1.1.2.2 SegMamba </b>          <!--   1.1.2.2 SegMamba   -->
   </summary>   
    
   The Paper: [SegMamba: Long-range Sequential Modeling Mamba For 3D Medical Image Segmentation](https://arxiv.org/pdf/2401.13560)

贡献：

- 整体架构使用的是U-Net的架构
- Mamba block改成了TSMamba Block，如图Fig.2里面的样子，里面涉及了一些模块
    - input x is [C,D,H,W]
    - x = GSC(x) = x + Conv3d_333(Conv3d_333(x) * Conv3d_111(X)), 每一个卷积都代表着 Norm->Conv3D->Nonlinear
    - x = x + LayerNorm(ToM(x))
        - ToM(x)为Mamba模块，其中有三个方向，如Fig.3b所示，forward，reverse和inter-wise，这个inter-wise代表的是竖着的
    - x = x + MLP(LayerNorm(x))
  

<img src="https://github.com/BaoBao0926/Paper_reading/blob/main/Image/1.Mamba/1.1%20VisionMamba/1.1.2%20Segmentation%20in%20medical%20image/SegMamba.png" alt="Model" style="width: 600px; height: auto;"/>

使用的数据集：

    - CRC-500: 文章提的
    
    - BraTS2023 Dataset

    - AIIB2023 Dataset

   <br />

</details>







</details>  <!--    -----------------------------------------  1.1.2  Segementation in Medical Image  ------------------------------------------  -->




   
</details>      <!--    -----------------------------------------  1.1 Vision Mamba   -------------------------------------------------------  -->

</details>      <!--    -----------------------------------------  1. Mamba   -------------------------------------------------------  -->
