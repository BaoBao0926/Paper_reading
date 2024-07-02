### Hi, I'am Muyi Bao

Here, I will put some paper about Vision Mamba used in medical image segmentation, more focusing on 3D segmentation.

<details>        <!-------------------------------------------------------------------   1.1.2.1  U-Mamba   ---------------------------------------------------------------------------->
   <summary>
   <b style="font-size: larger;">1.1.2.1 U-Mamba </b>         
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
- decodder是CNN-based的
  

<img src="https://github.com/BaoBao0926/Paper_reading/blob/main/Image/1.Mamba/1.1%20VisionMamba/1.1.2%20Segmentation%20in%20medical%20image/U-Mamba.png" alt="Model" style="width: 600px; height: auto;"/>

使用的数据集：

    - MICCAI 2022 FLARE Challenge
    
    - MICCAI 2022 AMOS Challenge
    
    - MICCAI 2017 EndoVis Challenge
    
    - NuerIPS 2022 Cell Segmentation Challenge

   <br />

</details>

<details>    <!---------------------------------------------------------------------------------    1.1.2.2 SegMamba  --------------------------------------------------------- -->
   <summary>
   <b style="font-size: larger;">1.1.2.2 SegMamba </b>       
   </summary>   
    
   The Paper: [SegMamba: Long-range Sequential Modeling Mamba For 3D Medical Image Segmentation](https://arxiv.org/pdf/2401.13560)

贡献：

- 整体架构使用的是U-Net的架构
- 第一层是Stem Convolutional Network, kernal size of 7 * 7 * 7, padding of 3 * 3 * 3 and stride of 2 * 2 * 2。在文章第一段提到，有一些工作为了提取large range information form 高分辨率3D图像，在一开始就会使用特别大的卷积核来促进感受野
- Mamba block改成了TSMamba Block，如图Fig.2里面的样子，里面涉及了一些模块
    - input x is [C,D,H,W]
    - x = GSC(x) = x + Conv3d_333(Conv3d_333(x) * Conv3d_111(X)), 每一个卷积都代表着 Norm->Conv3D->Nonlinear
       - 这个GSC(gated spatial convolution)，门控空间卷积模块，理论上可以增强在ToM之前空间维度上的特征表示
    - x = x + LayerNorm(ToM(x))
        - ToM(x)为Mamba模块，其中有三个方向，如Fig.3b所示，forward，reverse和inter-wise，这个inter-wise代表的是竖着的
        - ToM(x) = Mamba(x_forward) + Mamba(x_reverse) + Mamba(z_inter-slice)
    - x = x + MLP(LayerNorm(x))
- decoder是CNN-based的
  

<img src="https://github.com/BaoBao0926/Paper_reading/blob/main/Image/1.Mamba/1.1%20VisionMamba/1.1.2%20Segmentation%20in%20medical%20image/SegMamba.png" alt="Model" style="width: 800px; height: auto;"/>

使用的数据集：

    - CRC-500: 文章提的
    
    - BraTS2023 Dataset

    - AIIB2023 Dataset

   <br />

</details>


<details>    <!---------------------------------------------------------------------------------    1.1.2.2 SegMamba  --------------------------------------------------------- -->
   <summary>
   <b style="font-size: larger;">1.1.2.2 SegMamba </b>       
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































