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

    - CRC-500: 文章自己提出的
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

- 整体架构使用的是U-Net的架构,并且这是第一篇只采用的是纯SSM的结构，也就是decoder里面没有任何的卷积层，patch embeding用的4 * 4的patch size
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



<details>     <!---------------------------------------------------   1.1.2.10 H-VMUNet   ---------------------------------------------------------------------->
   <summary>
   <b style="font-size: larger;">1.1.2.10 H-VMUNet 2024/7/6 </b>         
   </summary>   
    
   The Paper, published in 2024.3.20: [H-vmunet: High-order Vision Mamba UNet for Medical Image Segmentation](https://arxiv.org/pdf/2403.13642)

   The official repository: [here](https://github.com/wurenkai/H-vmunet)
   

贡献：

- 这篇文章对于整个Mamba Block进行了修改(不是SSM)，是参考这篇工作【HorNet: Efficient High-Order Spatial Interactions with Recursive Gated Convolutions】修改的。其次对于skip connection part也进行了修改。
- [【CSDN Blog for HorNet: Efficient High-Order Spatial Interactions with Recursive Gated Convolutions】](https://blog.csdn.net/gaopursuit/article/details/126711612?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522172025984816800225580120%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=172025984816800225580120&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduend~default-1-126711612-null-null.142^v100^pc_search_result_base4&utm_term=Efficient%20high-order%20spatial%20interactions%20with%20recursive%20gated%20convolutions&spm=1018.2226.3001.4187):  大概意思就是，这篇文章的作者认为Transformer取得成功的主要原因是因为dot-product self-attention可以实现高境界特征交互(high-order spatial interatction)，与Fig.3c类似，作者通过门控和循环实现了基于卷积的高阶空间交互建模。
- 整体架构使用的是U-Net的架构,下采样用的卷积，encoder用了两个卷积层和四个H-VSS block，decoder也是H-VSS Block和两个卷积，SSM没有改动，使用的是VMamba里面的四方向，中间的skip connection part也进行了一些修改
  - Encoder部分：一共六层，前两层为卷积层，kernel size为3，后四层为H-VSS block层，并紧跟一个卷积层(我认为是用于下采样)
  - 连接的部分，与VM-UNet-v2类似，其实也是SAB spatial attention bridge和cab channel attention module，也就是注意力机制，这里图里面显示shared表示参数共享，是因为SAB里面由卷积 CAB里面由FC，这些参数是共享的
     - SAB(x) = x + x * Conv2d(k=7)([MaxPool(x); AvgPool(x)])
     - CAB(x) = x + x * Sigmoid(FC(GAP(x)))
  - Decoder部分: 与Encoder对称，由两层卷积和四层H-VSS Block层
- H-VSS Block构成
   - 首先，mamba用的是VMamba的四方向，架构用的是HorNet里面的架构，里面的Mul是Hadamard product operation
   - 主要想法就是，在一个block中，重复的不断进行N次ssm(N即为n-order的n),并且维度从小的不断变大，从而实现高层次到低层次的计算
   - 如Fig.3b的H3-SS2D所示，先通过一个映射层把维度扩展到2C，然后第一个用C/4，第二个用C/2，第三个用C维度，这样进行的
      - 有点不清楚的是，每一次维度的扩展只可能发生在SS2D里面，所以只有可能是Fig.1a里面的SSM里面的MLP进行扩维度
   - H-SS2D里面用到的Local-SS2D是一个卷积和ssm的结合，会维持维度不变

使用的数据集：

    - ISIC17, international skin imaging collaboration
    - Spleen dataset, sourced from Memorial Sloan Kettering Cancer Cente
    - Polyp dataset, MICCAI 2015 automated polyp detection subtask
    
    
<img src="https://github.com/BaoBao0926/Paper_reading/blob/main/Image/1.Mamba/1.1%20VisionMamba/1.1.2%20Segmentation%20in%20medical%20image/H-VMUnet.png" alt="Model" style="width: 1100px; height: auto;"/>


   <br />

</details>

<details>     <!---------------------------------------------------   1.1.2.11 Mamba-HUNet   ---------------------------------------------------------------------->
   <summary>
   <b style="font-size: larger;">1.1.2.11 Mamba-HUNet 2024/7/6 </b>         
   </summary>   
    
   The Paper, published in 2024.3.26: [Integrating Mamba Sequence Model and Hierarchical Upsampling Network for Accurate Semantic Segmentation of Multiple Sclerosis Legion](https://arxiv.org/pdf/2403.17432)

   The official repository: None
   
贡献：

- 看完前面的文章，这篇文章其实没有什么创新点，如果可以早三个月发，那还有点新意
- 唯一一个，而且也不是很重要的是，他们做了一个轻量级的模型，和一个网站，可以快速识别
- 纯Mamba结构，没有说下采样用的什么，有Patch embedding, 这也是Mamba-HUNet里面的H 代表hierarchy，也就是有层次的原因。
  

<img src="https://github.com/BaoBao0926/Paper_reading/blob/main/Image/1.Mamba/1.1%20VisionMamba/1.1.2%20Segmentation%20in%20medical%20image/Mamba-HUNet.png" alt="Model" style="width: 1100px; height: auto;"/>


   <br />

</details>





<details>     <!---------------------------------------------------   1.1.2.12 UltraLight VM-UNet   ---------------------------------------------------------------------->
   <summary>
   <b style="font-size: larger;">1.1.2.12 UltraLight VM-UNet 2024/7/6 </b>         
   </summary>   
    
   The Paper, published in 2024.3.29: [UltraLight VM-UNet:Parallel Vision Mamba Significantly Reduces Parameters for Skin Lesion Segmentation](https://arxiv.org/pdf/2403.20035)

   The official repository: [Here](https://github.com/wurenkai/UltraLight-VM-UNet)
   
贡献：

- 这篇文章做的最大贡献是模型轻量化，和前面看过的Light Mamba比起来，少了87%，只有0.049M的参数量和0.06GFLOPs，提出的PVM Layer是一个即插即用的模块，非常棒
- 整体架构使用的是U-Net的架构, 似乎没有下采样，Encoder用了3层ConV Block，然后三层PVM Layer，Decoder是对称的也是三层卷积，三层PVM Layer，中间的skip connection和前面看过的H-VMamba一样，使用的是SAB和CAB(spatical attention bridge和chanel attention bridge)
  - Encoder部分：一共六层，前3层为卷积层，kernel size为没有提，后3层为PVM Layer
  - 连接的部分，与H-VMamba一样子的，由共享参数的SAB和CAB组成
     - SAB(x) = x + x * Conv2d(k=7)([MaxPool(x); AvgPool(x)])
     - CAB(x) = x + x * Sigmoid(FC(GAP(x)))
  - Decoder部分: 与Encoder对称，由3层卷积和3层PVM Layer
- PVM Layer构成
   - 最核心想法如Fig.3所示，我们把channel划分成四份，对着每一份进行一个mamba的操作(从代码上看，都是同一个mamba)，这样可以节约非常多的参数，最后在拼到一起
   - 有一个Fig.4，我没有放到这里来，如果对着C channel数量的直接进行mamba，需要x个parameter，那么对着C/2进行两次mamba，只需要2*0.251(两个C/2是分开的mamba)，对着4 * C/4只需要0.063 * 4个参数。
   - 整体看起来非常简单，并且参数非常少，而且效果还不错，虽然不都是最好的，ISIC2017 DSC SE是最好的，PH^2全都是最好的，ISIC2018在DSC和ACC上是最好的
- 从代码中看到的实现细节
   - 首先关于CAB的实现，我们可以看到Fig.2里面的CAB其实有一个other stage，我以前没有看过对应的那篇文章。其实这里是6个stage的output要一起cat在一起，然后分别通过对应的linear层映射到各自的维度，所以这里其实是要把每一个stage的信息综合
   - 关于skip connection，从Fig.2上来看，每一个stage都要经过SAB CAB，但是事实上并不是，从代码来看stage 6并没有经过SAB CAB，甚至都没有skip connection，其实有点把stage当作bottleneck的感觉，这个绝对不是代码失误，因为上面提到CAB是吧所有stage合并在一起，但是代码其实只把前5个stage合并到一起了
   - PVM Layer中，四个分割开的channel group通过的其实是一个mamba block
   - 下采样用的是stride=2，size=2的maxpooling
   - encoder的卷积全部都是size=3,stride=1,padding=1的
   - decoder的最后一个卷积其实相当于segmentation head，输出num_class， size=1。 其他两个decoder size=3, stride=1, padding=1
  
使用的数据集：

    - ISIC2017
    - ISIC2018
    - PH^2，这是个什么external validation，不太理解什么意思


<img src="https://github.com/BaoBao0926/Paper_reading/blob/main/Image/1.Mamba/1.1%20VisionMamba/1.1.2%20Segmentation%20in%20medical%20image/UltraLight%20VM-UNet.png" alt="Model" style="width: 1100px; height: auto;"/>


   <br />

</details>




<details>     <!---------------------------------------------------   1.1.2.13 T-Mamba   ---------------------------------------------------------------------->
   <summary>
   <b style="font-size: larger;">1.1.2.13 T-Mamba 2024/7/7 </b>         
   </summary>   
    
   The Paper, published in 2024.4.01: [T-Mamba:Frequency-Enhanced Gated Long-Range Dependency for Tooth 3D CBCT Segmentation ](https://arxiv.org/pdf/2404.01065)

   The official repository: [Here](https://github.com/wurenkai/UltraLight-VM-UNet)
   
贡献：

- 这篇文章做的最大贡献是,把Frequency-domian引入计算，把每一个stage换成了类似于DenseNet的结构，最后在Mamba block里面加入了一个Gate Selection Unit,并且坚决加入位置编码，小创新点还是很多的，用于牙齿检测，所以叫做Tooth-Mamba
- 整体架构使用的是U-Net的架构(没有非常像), 没有说下采样是什么，只有三个stage，每一个stage都有5个unit的denseNet组成，然后跟着一个下采样，decoder好像只有一个predict head也没有别的了
  - Encoder部分：三个stage，每一个stage都有5个unit的denseNet组成，然后跟着一个下采样，由Tim Block组成
  - Decoder部分: 只有把三个stage的输出cat到一起，然后直接就是预测头了，估计也就是一个卷积处理了一下，所以和U-Net的那种构架有一些区别
- Tim的构成：
   - 首先，在进行token embeding的后面，这篇文章非常坚决的加入了位置编码，使用的是Transformer里面的那个sin cos位置编码方式，使用共享参数，在每一个Tim block里面都加上。这篇文章认为这样的位置编码可以保持spatial position保持不变， 同时减少模型的参数和计算负担，通过消融实验里面的数据，这个dual positional encoding的方式比signgle positional embedding的方式好
   - Frequency-based band pass filtering：大体就是引入F-Domian，
      - 在文中介绍到，高频率的成文会不中texture details，低频率的成分会encode shape informaton，这样，把frequency domain feature和spatial domain一起，可以宝成一个更准确的
      - 从图上看，也就是在Mambba的方向上，多引入了一个方向的分路，从公式(5)看起来，并没有用到ssm，只是很单纯的一个计算Bandpass的计算，有点没看懂这个bandpass是个什么东西
   - Gate selection Unit：在forward, backward和f-domian的输出后面通过这个gate selection unit去进行fusion的操作，主要操作在公式6和Fig.1c里面展示
      - 有一个小点，通过这篇[CSDN Blog](https://blog.csdn.net/weixin_43301333/article/details/114394629?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522172032419116777224487805%2522%252C%2522scm%2522%253A%252220140713.130102334.pc%255Fall.%2522%257D&request_id=172032419116777224487805&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~first_rank_ecpm_v1~rank_v31_ecpm-2-114394629-null-null.142^v100^pc_search_result_base4&utm_term=FC%E6%98%AF%E6%8C%87%E4%BB%80%E4%B9%88%EF%BC%8C%E5%9C%A8%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E4%B8%AD&spm=1018.2226.3001.4187),从Fig.1c，可以发现，Embedding Sequence->Adaptiva Pooling->MLP->FC,已经有了MLP干嘛还来一个FC呢？从blog看到，MLP一般指多层线性层 有hidden layer的，FC指fully connected layer，可以是单层可以是多层，所以我觉得可能是吧最后一层单独拿出来以作说明。


使用的数据集：

    - 3D CBCT dataset

<img src="https://github.com/BaoBao0926/Paper_reading/blob/main/Image/1.Mamba/1.1%20VisionMamba/1.1.2%20Segmentation%20in%20medical%20image/T-Mamba.png" alt="Model" style="width: 1100px; height: auto;"/>


   <br />

</details>





<details>     <!---------------------------------------------------   1.1.2.14 AC-MambaSeg   ---------------------------------------------------------------------->
   <summary>
   <b style="font-size: larger;">1.1.2.14 AC-MambaSeg 2024/7/7 </b>         
   </summary>   
    
   The Paper, published in 2024.5.09: [AC-MAMBASEG:AN ADAPTIVE CONVOLUTION AND MAMBA-BASED ARCHITECTURE FOR ENHANCED SKIN LESION SEGMENTATION ](https://arxiv.org/pdf/2404.01065)

   The official repository: [Here](https://github.com/vietthanh2710/AC-MambaSeg)
   
贡献：

- 这篇文章把CBAM, Attention Gate, Selective Kernel Bottleneck都结合到了一起
- 整体架构使用的是U-Net的架构, 似乎没有下采样，有5个stage，每一个stage由ResVSS Block组成。skip connection由CBAM组成，Decoder和Encoder一样
  - Encoder部分：有5个stage，每一个stage由ResVSS Block组成，具体组成看Fig.3
  - Decoder部分: 有5个stage，每一个stage由ResVSS Block组成，但是decoder和encoder是不一样的，具体组成看Fig.4
     - skip connection送过来的部分还经过了一个Attention Gate，
  - Skip Connection部分：使用了CBAM做注意力机制
  - BottleNeck：使用了Selective Kernel和点积，Selective Kernel可以参考这个[CSDN Blog](https://blog.csdn.net/qq_52302919/article/details/123078832?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522172032469316800178588516%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=172032469316800178588516&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~baidu_landing_v2~default-1-123078832-null-null.142^v100^pc_search_result_base4&utm_term=%20Selective%20Kernels%20Convolution&spm=1018.2226.3001.4187)
- ResVSS Block：
   - ResVSS(x) = x * Scale + DWConv(VSS(InstanceNorm+ReLU(x)))
   - VSS使用的是VMamba里面的四个方向的



使用的数据集：

    - ISIC-2018
    - PH2

<img src="https://github.com/BaoBao0926/Paper_reading/blob/main/Image/1.Mamba/1.1%20VisionMamba/1.1.2%20Segmentation%20in%20medical%20image/AC-MambaSeg1.png" alt="Model" style="width: 1100px; height: auto;"/>

<img src="https://github.com/BaoBao0926/Paper_reading/blob/main/Image/1.Mamba/1.1%20VisionMamba/1.1.2%20Segmentation%20in%20medical%20image/AC-MambaSeg2.png" alt="Model" style="width: 1100px; height: auto;"/>

<img src="https://github.com/BaoBao0926/Paper_reading/blob/main/Image/1.Mamba/1.1%20VisionMamba/1.1.2%20Segmentation%20in%20medical%20image/AC-MambaSeg3.png" alt="Model" style="width: 500px; height: auto;"/>

   <br />

</details>








<details>     <!---------------------------------------------------   1.1.2.15 HC-Mamba   ---------------------------------------------------------------------->
   <summary>
   <b style="font-size: larger;">1.1.2.15 HC-Mamba 2024/7/7 </b>         
   </summary>   
    
   The Paper, published in 2024.5.08: [HC-MAMBA: VISION MAMBA WITH HYBRID CONVOLUTIONALTECHNIQUES FOR MEDICAL IMAGE SEGMENTATION](https://arxiv.org/pdf/2405.05007)

   The official repository: None
   
贡献：

- 这篇文章把空洞卷积和channel shuffle应用到了一起，算是第一篇把空洞卷积应用到Mamba里面的文章(至少在segmentation里面)
- 整体架构使用的是U-Net的架构, 下采样为patch merging，有4个stage，每一个stage由两个HC-SSM Block组成。skip connection直接连，Decoder和Encoder一样
  - Encoder部分：有4个stage，每一个stage由HC-SSM Block组成，具体组成在Fig.2b,也就是HC-SSM Block
  - Decoder部分: 有4个stage，与Encoder对称
- HC-SSM Block：如图Fig.2b
   - 开头的X的channel会被分成两份X1,X2
      - X1进行他们这边提出的卷积模块，其实也就是dilation convolution空洞卷积，HC-Conv block，先一个空洞卷积，然后跟着一个Depthwise Separable Conv
      - X2进行SSM，用的是VMamba里面的四方向的扫描
   - 把X1和X2拼接到一起了之后，进行channel shuffle，参考[CSDN Blog](https://blog.csdn.net/weixin_43334693/article/details/130905826?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522172033138416800184137617%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=172033138416800184137617&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduend~default-1-130905826-null-null.142^v100^pc_search_result_base4&utm_term=channel%20shuffle%20operation%2C&spm=1018.2226.3001.4187), 我的理解是，cat之后是[HC-Conv Branch; SSM-Branch], 然后把他变成了HC-Conv-Branch第一个channel，然后是SSM-Branch的第一个Channel，然后是Hc-Conv-Branch第二个channel，然后是SSM-Branch的第二个channel，让channle混合在一起了
  - 最后有一个残差


使用的数据集：

    - ISIC-2017
    - ISIC-2018
    - Synapse


<img src="https://github.com/BaoBao0926/Paper_reading/blob/main/Image/1.Mamba/1.1%20VisionMamba/1.1.2%20Segmentation%20in%20medical%20image/HC-Mamba.png" alt="Model" style="width: 500px; height: auto;"/>

   <br />

</details>





<details>     <!---------------------------------------------------   1.1.2.16 UU-Mamba   ---------------------------------------------------------------------->
   <summary>
   <b style="font-size: larger;">1.1.2.16 UU-Mamba 2024/7/7 </b>         
   </summary>   
    
   The Paper, published in 2024.5.08: [UU-Mamba:Uncertainty-aware U-Mamba for Cardiac Image Segmentation](https://arxiv.org/pdf/2405.17496)

   The official repository: None
   
贡献：

- 这篇文章不是很有看头，用的是U-Mamba的结构，VMamba的SS2D的ssm模块，Mamba block略微改了一点，主要点在于，它提出了使用Uncertianty-aware Loss进行训练，使用SAM optimization进行优化，
- 整体架构使用的是U-Net的架构, 与UMamba保持一致，下采样为Conv，有4个stage，每一个stage由。skip connection直接连，Decoder和Encoder一样，看看图就知道了
- Uncertainty-Aware Loss: 文章提出，传统的Cross-Entropy loss可能不能处理好class imbalance or the fine details，它这种Uncertainty-Aware Loss可以做到engance resilience and precision by focusing on confident predictions and reducing the influence of ambiguous ones
   - Dice Coefficient (DC)loss: 这种基于区域的度量强调了预测和地面真值区域之间的重叠，保持了分割区域的形状和边界的准确性This region-based metric em phasizes the overlap between predicted and ground truth areas,preserving the accuracy of the shape and boundaries of segmented regions.
   - Cross-Entropy (CE)loss: 这种基于分布的损失确保了对单个像素的准确分类，从而提高了分类精度。This distribution-based loss ensures accurate categorization of individual pixels,im proving classification precision.
   - Focal loss: 这种像素级损失通过对难以分类的情况赋予更大的重要性来解决类不平衡问题，增强了模型处理复杂场景的能力 This pixel-level loss addresses class imbalance by assigning greater importance to difficult-to-classify in stances,enhancing the model’s capability to handle com plex scenarios
- 文章用了Sharpness-Aware Minimization (SAM) optimization优化器进行训练，我没有细看这一部分，但是我觉得既然大部分的文章是用Adam或者AdamW的，一定是由道理的


使用的数据集：

    - ACDC dataset, Automated Cardiac Diagnosis Challenge


<img src="https://github.com/BaoBao0926/Paper_reading/blob/main/Image/1.Mamba/1.1%20VisionMamba/1.1.2%20Segmentation%20in%20medical%20image/HC-Mamba.png" alt="Model" style="width: 1000px; height: auto;"/>

   <br />

</details>



<details>     <!---------------------------------------------------   1.1.2.17 SAM-VMNet   ---------------------------------------------------------------------->
   <summary>
   <b style="font-size: larger;">1.1.2.17 SAM-VMNet 2024/7/7 </b>         
   </summary>   
    
   The Paper, published in 2024.5.08: [SAM-VMNet:Deep Neural Networks For Coronary Angiography Vessel Segmentation](https://arxiv.org/pdf/2406.00492)

   The official repository: None
   
贡献：

- 这篇文章不是很创新，不过我也不知道究竟怎么样，但是把SAM(Segment Anything in Medical Images)这篇工作的内容拿过来了和Mamba一起进行了合作，算是第一篇Mamba和SAM的融合，但是我感觉融合的不是很好。
- 我没有看过关于SAM的内容，所以我也不太清楚里面的工作是什么，简单来看，有提示工程(有点像是clip了),然后医学图像的数据集很大，好像有100w
- 整体架构使用的是U-Net的架构
   - 关于MedSAM的部分，是这样的，通过一个训练好了的VM-UNet进行一次粗预测，得到的预测结果和原图一起扔到MedSAM的encoder里面进行的东西，这一部分的训练参数全部冻结
   - 上面得到的东西和Encoder cat到一起，然后进行decoder的预测

使用的数据集：

    - ARCADE dataset


<img src="https://github.com/BaoBao0926/Paper_reading/blob/main/Image/1.Mamba/1.1%20VisionMamba/1.1.2%20Segmentation%20in%20medical%20image/SAM-VMNet.png" alt="Model" style="width: 1000px; height: auto;"/>

   <br />

</details>



<details>     <!---------------------------------------------------   1.1.2.18 MHS-VM   ---------------------------------------------------------------------->
   <summary>
   <b style="font-size: larger;">1.1.2.18 MHS-VM 2024/7/7 </b>         
   </summary>   
    
   The Paper, published in 2024.6.10: [MHS-VM: Multi-Head Scanning in Parallel Subspaces for Vision Mamba](https://arxiv.org/pdf/2406.00492)

   The official repository: [here](https://github.com/PixDeep/MHS-VM)
   
贡献：

- 这篇文章对于Mamba里面的scan pattern和scan route进行了探究，并提出了一个fusion的策略，感觉可以作为一个即插即用的模块
- 整体架构使用的是VMamba的框架，只是把VMamba的SS2D换成了这篇文章的Multi-Head Scan modules MHS module
- 这里开始写MHS module，有一些地方写的不是很清楚，但是我又没有看代码，所以有些地方只能比较猜测的去写
   - 从Fig.1的最下面开始看，Head部分，首先x被映射到不同的分路里面，也是为什么这篇文章会叫自己parallel subspaces的原因，一方面他们是并行的，一方面映射到不同的分路之后，相当于不同的子空间进行计算，类似于Transofrmer，维度是要变小的，如果原来的x的Channel为C，那么对于n个子空间映射之后变成了C/n维度
   - 然后经过一个Scan Head,这里写的不太清楚，
      - 我们可以知道一共有4种scan pattern如Fig.2，有4种Scan routes如Fig.4，
      - 在现在结构中，一个subspace(一条分路)只使用一个scan pattern， in our current architecture, one scan pattern is deployed per subspace
      - 文中提到每一个scan head默认使用四个routes，然后把四个routes stack到一起，但是这个Stack根据我得猜测和Fig.2(我没有放进来),我认为不是把这四个按照前后拼到一起，应该是相当于多了一个维度那么拼到一起的
      - 文中提到每一个Mamba block不共享参数，我认为是每一个子空间的mamba都是独立出来的，但是对于一个子空间，用的应该是一个，不然参数不会这么小(猜的)
      - 但是让我最不清楚的地方在于，这里有四种pattern，但是默认是只有三个scan head，那应该怎么分配？我没有看到任何描述
   - 对于一个pattern出来的四个routes的结果，丢到ESF(Embedding Section Fusion)里面去，进行一个fusion，这个文章提供了四种fusion的方式，进行了比较，但是第四种方式直接在讲述的时候就被pass掉了
      - 第一种方式Direct Addition，把所有的直接加在一起
      - 第二种方式Mixture of Pooling，如Fig.5a所示，是CBAM的操作，Linear([AvgPool(x);MaxPool(x)])
      - 第三种方式CV-GUIDED Scaling, CV是Coefficient of Variation,具体公式如下图所示: <img src="https://github.com/BaoBao0926/Paper_reading/blob/main/Image/1.Mamba/1.1%20VisionMamba/1.1.2%20Segmentation%20in%20medical%20image/MHS-VM2.png" alt="Model" style="width: 800px; height: auto;"/>
         - 变异系数是对样本或总体中平均值周围数据的相对可变性或分散性的度量(Coefficient of variation is a measure of relative variability or dispersion of data around the mean value in a sample or population)
         - 如果数据分散，那么CV就会大，在一定程度上，由这些扫描路径提取的嵌入很好地知道位置信息，并且值得保留或增强(To a certain extent,the embedding extracted by these scan routes is well aware of positional information and is worth preserved or enhancing)
         - 并且，从公式(9)中可以看到，这篇文章对于x(没有说清楚x究竟是谁)，目的是在图中写了，用一个参数t来filter out feature with lower CV value(这说明希望要一个高CV的结果)，后面也论述到这可以作为一种新的regularization的方法阻止overfitting和generalization
      - 第四种方式，可以把第二种方式和第三种方式结合起来， z4 = z2 (*) monotone(ycv)
         - z4，z2代表第几种方式的结果，z2就是第二种方式的结果，z4就是第四方式的结果
         - (*)代表的是element-wise product，github不知道怎么打出对应的符号
         - montone代表的是activation function，such as Sigmoid and ReLU
         - ycv可以从上图的公式8的下面第一行看到 ycv的公式，ycv= std([yi[)/avg([yi-min([yi])])
         - 但是文章讨论说，在ISIC18上的舒颜表明，这个没有significant performance improvement，并且会增加计算复杂度，所以就被pass了
- 消融实验的结果
   - ESF fusion的结果是，CV-guided Scaling > Mixture of Pooling > Directly Sum
   - Fig.1里面的最后tail里面的projection，结果是有Projection会更好，毕竟更深了
   - Number of scan head: 4个子空间(子空间维度C/3) > 4个子空间(子空间维度C/4) > 3个子空间(子空间C/3)， 所以子空间变多，维度变多是有一定作用的，但是后两个的参数量差不多都是在17M，第一个的参数量是20M，变化了很多
   - 最后是于VM-UNet和VM-UNet-T进行了结果比较，因为这篇文章相当于提出了一个plug-and-play的模块，所以只把VM-UNet里面的SS2D给替换了，性能大概可以表述为比VM-UNet-T差不多，比VM-UNet基本一致，其中VM-UNet-T是使用了VM-UNet的权重作为预训练权重然后继续训练的结果

     
使用的数据集：

    - ISIC17
    - ISIC18
    - Synapse


<img src="https://github.com/BaoBao0926/Paper_reading/blob/main/Image/1.Mamba/1.1%20VisionMamba/1.1.2%20Segmentation%20in%20medical%20image/MHS-VM.png" alt="Model" style="width: 1000px; height: auto;"/>

   <br />

</details>








