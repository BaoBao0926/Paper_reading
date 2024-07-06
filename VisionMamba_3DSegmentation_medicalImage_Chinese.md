### Hi, I'am Muyi Bao

Here, I will put some paper about Vision Mamba used in medical image segmentation, more focusing on 3D segmentation.

---

好多文章都会提到：
- CNN-based方法对于局部和全局的感受野会受限
- Transformer有了全局视野，但是需要heavy computational load，在面对高维高分辨率的图像的时候

---

<details>        <!-------------------------------------------------------------------   1.1.2.1  U-Mamba   ---------------------------------------------------------------------------->
   <summary>
   <b style="font-size: larger;">1.1.2.1 U-Mamba 2024/6/30 </b>         
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
   <b style="font-size: larger;">1.1.2.2 SegMamba 2024/6/30 </b>       
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


<details>    <!---------------------------------------------------------------------------------    1.1.2.3 nnMamba  --------------------------------------------------------- -->
   <summary>
   <b style="font-size: larger;">1.1.2.3 nnMamba 2024/7/1 </b>       
   </summary>   
    
   The Paper: [nnMamba: 3D Biomedical Image Segmentation, Classification and Landmark Detection with State Space Model](https://arxiv.org/pdf/2402.03526)

贡献：

- 这篇文章其实也算是一个通用骨架了，但是没有非常通用，对面classification和dense prediction的时候会有对应的修改
- 整体架构使用的是U-Net的架构
- Segmentation and Landmark Detection架构
   - StemConv->ResMamba->ResMamba->ResMamba->Double Conv->Double Conv->Double Conv
      - StemConv应该是大卷积核
      - Res-Mamba是 x = x + Relu(BN(Conv3 * 3 * 3(Relu(BN(Conv3 * 3 * 3(x)))))) + miccai(x)
      - miccai是这篇文章提出来的一个模块，实际上分为两个部分，MIC和CAI
         - MIC，Mamba in Convolution，这个模块通过Network-In-Network而启发的
            - 让ConvMIC(x) = Relu(BN(Conv1 * 1 * 1(x)))
            - x = ConvMIC(ConvMIC(x) + CAI(ConvMIC(x)))
         - CAI, Channel and Spatial with Siamese Input, 这个是被用于MIC里面的一个模块
            - 如图Fig.2e所示，就是一个四通道的SSM，有flip channel，flip length，flip channel&length和original
- Classification的架构：
   - 整体如Fig.2b所示，应该是每一个ResBlock的输出一起经过一个Average pooling, 然后得到的经过一个MICCAI，一个MLP，通过MLP进行预测 

  

<img src="https://github.com/BaoBao0926/Paper_reading/blob/main/Image/1.Mamba/1.1%20VisionMamba/1.1.2%20Segmentation%20in%20medical%20image/nnMamba.png" alt="Model" style="width: 800px; height: auto;"/>

   <br />

</details>


<details>    <!---------------------------------------------------------------------------------    1.1.2.4 VM-UNet  --------------------------------------------------------- -->
   <summary>
   <b style="font-size: larger;">1.1.2.4 VM-UNet 20244/7/1 </b>       
   </summary>   
    
   The Paper: [VM-Unet: Vision Mamba UNet for Medical Image Segmentation](https://arxiv.org/pdf/2402.03526)

贡献：

- 整体架构使用的是U-Net的架构,并且这是第一篇只采用的是纯SSM的结构，也就是decoder里面没有任何的卷积层
- 这篇文章叫自己Vision Mamba，但实际上使用的是VMamba厘米那的模块VSS block，进行了一定的修改，如Fig.1
   - SSM采用的是VMamba里面的四个扫描方向，forward，reverse和竖着的forward和reverse
- 似乎对着Loss function进行了一定的探究在section3.3，但是好像不是很关键

<img src="https://github.com/BaoBao0926/Paper_reading/blob/main/Image/1.Mamba/1.1%20VisionMamba/1.1.2%20Segmentation%20in%20medical%20image/VM-Unet.png" alt="Model" style="width: 800px; height: auto;"/>

   <br />

</details>




<details>     <!---------------------------------------------------   1.1.2.5 Swin-UMamba   ---------------------------------------------------------------------->
   <summary>
   <b style="font-size: larger;">1.1.2.5 Swin-UMamba 2024/7/4 </b>         
   </summary>   
    
   The Paper, published in 2024.2.5: [Swin-UMamba: Mamba-based UNet with ImageNet-based pretraining](https://arxiv.org/pdf/2402.03302)

贡献：

- 整体架构使用的是U-Net的架构，想要模仿Swim-Transformer的做法(但是这里存在一些问题，我在下面提到了)
- encoder部分使用的是VMamba的VSS block，decoder使用了mamba-based和CNN-based两者
   - mamba-based decoder的计算量和参数量要明显少于CNN-based decoder。在面对AbdomenMRI数据集的时候:
      - parameter数量从CNN的60M降到了28M
      - FLOPs从69G降到了18.9G
   - 性能表现依赖于数据集
      - AbdomeMRT数据集，mamba-based decoder会更好
      - Endoscopy数据集和Microscopy数据集，CNN-based decoder会更好， 
- 使用了deep supervision的策略，[paper: Deeply-Supervised Nets](https://proceedings.mlr.press/v38/lee15a.pdf) 和可参考的[CSDN blog](https://blog.csdn.net/qq_40507857/article/details/121025445?ops_request_misc=&request_id=&biz_id=102&utm_term=deep-supervised%20net&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduweb~default-0-121025445.142^v100^pc_search_result_base4&spm=1018.2226.3001.4187)。其实也就是在decoder的一些(该文章是3个)隐藏层中进行最终任务的分割，造成一些损失，从而加速训练
- 这篇文章称自己为第一篇探究了mamba-based model的关于预训练的性能提升。似乎是在之前的一些文章(CNN-based和Transformer-based)使用大分类数据集进行与训练可以提升性能，但是mamba-based model大多数还是从头开始训练(我猜也有训练变快了的原因),所以这篇文章探究了现在ImageNet上进行预训练，然后在进行分割任务。
   - ImageNet-based pretraining可以提升很多的性能，比如面对AbdomenMRI Dataset的时候，可以提升7%的性能

对于Swin这个点，我有一些想法。这篇文章其实想要模仿的是Swim-Transformer的结构，包括VM-UNet其实也有一点想要模仿的意思。但是对于Swin而言，我认为最重要的有两点：

- 第一点是Swin的w window，也就是我们要在一个widow里面进行自注意力，如果要用到mamba里面，那我们应该要对一个widow里面的patch进行ssm操作才对。才更加符合window这个概念，但是按照Swin-Transformer里面的参数，一个window有7 * 7个patch，49个patch对于Mamba来说可能有一点太短了(有可能，我也不确定，毕竟mamba号称可以处理百万序列）
- 第二个点是Swin的s shift，也就是为了量window之间有信息交互，所以要进行shift，那么这篇文章也就没有对这个进行处理
- 所以事实上，这篇文章只是模仿了Swin-Transoformer的patch merging而已，我认为没有使用到Swin里面的最核心的观念S和W。

<img src="https://github.com/BaoBao0926/Paper_reading/blob/main/Image/1.Mamba/1.1%20VisionMamba/1.1.2%20Segmentation%20in%20medical%20image/Swin-UMamba.png" alt="Model" style="width: 600px; height: auto;"/>

使用的数据集：

    - AdbomenMRI, MICCAI 2022 AMOS Challenge
    
    - Endoscopy, MICCAI 2017 EndoVis Challenge
    
    - Microscopy, NuerIPS 2022 Cell Segmentation Challenge

   <br />

</details>


<details>     <!---------------------------------------------------   1.1.2.6 Mamba-UNet   ---------------------------------------------------------------------->
   <summary>
   <b style="font-size: larger;">1.1.2.6 Mamba-UNet 2024/7/4 </b>         
   </summary>   
    
   The Paper, published in 2024.2.7: [Mamba-UNet: UNet-Like Pure Visual Mamba for Medical Image Segmentation](https://arxiv.org/pdf/2402.05079)

没有什么创新，没有太多价值

贡献：

- 整体架构使用的是U-Net的架构,下采样用的patch merging，纯Mamba block，没有用到卷积
- encoder和decoder都使用的是VMamba的VSS block
- 和前面文章比起来，这篇工作没有太多创新，就是把VMamba拿过来直接用，前面的文章好歹还会改一点mamba block之列的
  

<img src="https://github.com/BaoBao0926/Paper_reading/blob/main/Image/1.Mamba/1.1%20VisionMamba/1.1.2%20Segmentation%20in%20medical%20image/Mamba-UNet.png" alt="Model" style="width: 600px; height: auto;"/>

使用的数据集：

    - ACDC MRI cardica segmentation dataset, Automated Cardiac Diagnosis Challenge
    
    - Synpse multi-organ segmentation Challenge, MICCAI 2015 Multi-Atlas Abdomen Labeling Challenge
    

   <br />

</details>



<details>     <!---------------------------------------------------   1.1.2.7 LightM-UNet   ---------------------------------------------------------------------->
   <summary>
   <b style="font-size: larger;">1.1.2.7 LightM-UNet 2024/7/5 </b>         
   </summary>   
    
   The Paper, published in 2024.3.8: [LightM-UNet: Mamba Assists in Lightweight UNet for Medical Image Segmentation](https://arxiv.org/pdf/2403.05246)

贡献：

- 这篇文章相当于第一篇进行Mamba-based分割任务的参数优化的文章，压缩的相当狠，从U-Mamba的173M压缩到了1.87M，并且性能还有一点提升
- 整体架构使用的是U-Net的架构,下采样用的Max Pooling，纯Mamba block(有一点 点积DWConv)，为了节约参数，decoder部分没有正儿八经的模块，只用了一个点积而已
  - Encoder部分：DWConv->Encoder Block->Encoder Block->Encoder Block->Bottleneck Block
     - Encoder Block: 对于第l个encoder，先经过l个RVM Layer，最后一个RVM Layer会增加channel数量，然后经过一个max-pooling，降低resolution
     - RVM Layer(x) = Projection(LayerNorm(Scale*x + VSS(LayerNorm(x))))
     - VSS为Vision Mamb的block，forward和backword的那个
  - Decoder部分，很多个Decoder Block堆叠，每一个Block都是固定的
     - Decoder(x) = Interpolation(relu(Scale*x + DWConv(x))), Interpolation为bilinear interpolation， x为上一层的输出和残差连接的输出之和

想法：

考虑到前面的Swin-UMamba里面提到的，使用Mamba作为decoder可以减少大量的参数而言，如果直接把decoder的复杂卷积全部抛弃，事实上确实有希望让参数变的非常少非常少，并且把下采样换成了maxpooling，感觉有点奇怪，但是好像也可以说的过去。但是让我很惊讶的是，性能还有有一定的提升，这是和U-Mamba比较的。

使用的数据集：

    - LiTs dataset， 3D CT image
    
    - Montogomery&Shenzhen dataset, 2D X-ray images

<img src="https://github.com/BaoBao0926/Paper_reading/blob/main/Image/1.Mamba/1.1%20VisionMamba/1.1.2%20Segmentation%20in%20medical%20image/LightM-UNet.png" alt="Model" style="width: 600px; height: auto;"/>


    

   <br />

</details>



<details>     <!---------------------------------------------------   1.1.2.8 LKM-UNet   ---------------------------------------------------------------------->
   <summary>
   <b style="font-size: larger;">1.1.2.8 LKM-UNet 2024/7/6 </b>         
   </summary>   
    
   The Paper, published in 2024.3.12: [Large Window-based Mamba UNet for Medical Image Segmentation: Beyond Convolution and Self-attention](https://arxiv.org/pdf/2403.07332)

   The official repository: [here](https://github.com/wjh892521292/LKM-UNet)

贡献：

- 这篇文章对于mamba的输入而言做了修改，第一个(PiM)是在一个winodw里面的所有像素的ssm，第二个(PaM)是对着这个widow进行pooling，然后对着pooling之后的所有window进行ssm。前者实现local scope pixel之间的信息交互，避免遗忘了邻近区域内部的信息，后者实现long-range dependency modeling and global patch interaction
- 整体架构使用的是U-Net的架构,下采样用的没说，decoder为卷积，使用的是Vim里面的双向
  - Encoder部分：先一个Depth-wise Conv,然后就是四层LM Block(由一个PiM和一个PaM组成)
     - PiM为pixel-level SSM: 把input image划分为window，在一个window内部，对着所有的像素进行mamba操作
        - 从文章的消融实验来看，如果这个window的size变大，性能反而提升
     - PaM为patch-level SSM：把经过PiM的输出进行一次pooling(没有说什么pooling)，然后一个window就相当于一个token了，对着所有的window进行mamba操作，最后来一个Unpooling
     - PiM的输出和PaM的输出通过残差相加
  - Decoder部分，就是卷积的输出，类似于ViT那种的，也没有详细介绍


使用的数据集：

    - Adbomen CT, MICCAI 2022 FLARE Challenge
    
    - Adbomen MR, MICCAI 2022 AMOS Challenge
    
<img src="https://github.com/BaoBao0926/Paper_reading/blob/main/Image/1.Mamba/1.1%20VisionMamba/1.1.2%20Segmentation%20in%20medical%20image/LKM-UNet.png" alt="Model" style="width: 600px; height: auto;"/>


   <br />

</details>







<details>     <!---------------------------------------------------   1.1.2.9 VM-UNet-V2   ---------------------------------------------------------------------->
   <summary>
   <b style="font-size: larger;">1.1.2.9 VM-UNet-V2 2024/7/6 </b>         
   </summary>   
    
   The Paper, published in 2024.3.12: [VM-UNET-V2: Rethinking Vision Mamba UNet for Medical Image Segmentation](https://arxiv.org/pdf/2403.09157)

   The official repository: [here](https://github.com/nobodyplayer1/VM-UNetV2)
   

贡献：

- 这篇文章对于Encoder和Decoder之间的skip connection做了修改
- 这篇文章应该是参考的了这篇文章 【U-net v2:Rethinking the skip connections of u-net for medical image segmentation】，因为名字都差不多，而且文中提到了这篇文章，结构也差不多。从这篇文章参考资料, 里面用到了这篇文章【Cbam:Convolutional block attention module】的内容，不是VM-UNet-v1的作者写的。
   - [UNet-v2 CSDN Blog](https://blog.csdn.net/qq_29788741/article/details/134796090?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522172024792516800182168790%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=172024792516800182168790&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduend~default-1-134796090-null-null.142^v100^pc_search_result_base4&utm_term=Unet-v2&spm=1018.2226.3001.4187): 从Unet-v2来看，就是对于skip connection进行了一些处理,使用到了CBAM里面的attention module(不是transformer的自注意力机制)，让每一个stage输出的特征图进行进行注意计算，然后使用dowsample让特征图大小一样，最后使用Hadamard product(这个就是矩阵中对应位置的元素相乘,参考[CSDN Blog](https://blog.csdn.net/qq_42363032/article/details/122538639?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522172024489316800227419590%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=172024489316800227419590&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_click~default-1-122538639-null-null.142^v100^pc_search_result_base4&utm_term=Hadamard%20product&spm=1018.2226.3001.4187))，把所有处理之后的特征图相乘。
   - [CBAM CSDN Blog](https://blog.csdn.net/m0_45447650/article/details/123983483?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522172024715916800184118767%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=172024715916800184118767&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_positive~default-1-123983483-null-null.142^v100^pc_search_result_base4&utm_term=Cbam&spm=1018.2226.3001.4187): 就是结合了通道和空间注意力机制模块
      - CAM channel attention module，通道维度不变，压缩空间维度，也就是C * H * W -> C * 1 * 1, 这代表了对于每一个channel的注意力。CAM(x) = activation(MLP(AvgPool(x)) + MLP(MaxPool(x)))
      - SAM spatial attention module, 空间维度不变，压缩通道位数，也就是C * H * W -> 1 * H * W, 这代表了对于目标的位置信息的关注, SAM(x) = activation(f([AvgPool(x); MaxPool(x)]))。
         - f代表7 * 7的卷积，实验表明7 * 7的比3 * 3的好，
         - 中间的操作是把avgpool和maxpool的输出拼接到一起
      - CBAM为CAM和SAM的结合，对于并行还是串行都有实验，结果是先通道再空间会好一点
- 整体架构使用的是U-Net的架构,下采样用的patch merging，decoder为卷积，使用的是Vim里面的双向
  - Encoder部分：先一个Depth-wise Conv,然后就是四层LM Block(由一个PiM和一个PaM组成)
  - 连接的部分，SDI模块，从图来看，先行过CBAM里面的注意力机制的修改，这样feature map的大小是不变的，然后通过下采样，变成最小的那个feature map的大小，然后使用Hadamard prodct得到输出，大小为最小的feature map的大小
  - Decoder部分，就是卷积的输出也没有详细介绍
     - 使用了deep surpervision，对于最后两个stage进行
     - fusion block说的有点模糊不清楚，因为SDI模块看起来输出的每次都是最小的feature map的大小，所以这样每一次的fusion block都是与最小的feature map大小进行的，所以感觉有点奇怪，可能具体要看代码才行。


使用的数据集：

    - ISIC17, ISIC18, CVC-300, CVC-ClinkcDB, Kvasir, CVC-ColonDB and ETIS-LaribPolypDB
    
    
<img src="https://github.com/BaoBao0926/Paper_reading/blob/main/Image/1.Mamba/1.1%20VisionMamba/1.1.2%20Segmentation%20in%20medical%20image/VM-UNet-V2.png" alt="Model" style="width: 1100px; height: auto;"/>


   <br />

</details>





