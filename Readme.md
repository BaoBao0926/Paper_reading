## Hi, I am [Muyi Bao](https://github.com/BaoBao0926/BaoBao0926.github.io)

[English](https://github.com/BaoBao0926/Paper_reading) | [简体中文](https://github.com/BaoBao0926/Paper_reading/blob/main/Readme_Chinese)

---

I am going to write somethings about the papers I read in this repository. I may reproduced some papers in this [repository](https://github.com/BaoBao0926/Overview-of-Reproduced-Project)

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

  Very similar to the Vision Transformer
   
   Contributions:

   - The scan direction is bi-directional-forward and backward. The backword direction is implemented by flip()
   - Use zero, one and two cls tokens:
     - zero cls token, use max, mean or other ways
     - one cls token, we can insert use token at head, tail, middle or random position
     - two cls tokens, inset on at head and other one at tail

<img src="https://github.com/BaoBao0926/Overview-of-Reproduced-Project/blob/main/Code/009.Vision%20Mamba(Vim)/architecture.png" alt="Model" style="width: 600px; height: auto;"/>
    
   <br />

</details>


  <details> 
   <summary>
   <b style="font-size: larger;">1.1.1.2 Visual Mamba(VMamba) </b>    <!--   1.1.1.2  Visual Mamba(VMamba)   -->
   </summary>   
    
   The Paper: [VMamba: Visual State Space Model](https://arxiv.org/abs/2401.10166)

  Very similar to the Swim Transformer
   
   Contributions:

   - 2D Selective Scan(SS2D): use four directions to scan and then merge
   - VSS Block: change Mamba block. see in figure.
   - Architecutre: Using hierarchial architecture(similar to swim transformer), patch number smaller 4 times and channel bigger 2 times 

<img src="https://github.com/BaoBao0926/Paper_reading/blob/main/Image/1.Mamba/1.1%20VisionMamba/1.1.1%20Backbone_network/VMamba.png" alt="Model" style="width: 600px; height: auto;"/>
    
   <br />

</details>


<details> 
   <summary>
   <b style="font-size: larger;">1.1.1.3 Mamba-ND </b>          <!--   1.1.1.3  Mamba-ND   -->
   </summary>   
    
   The Paper: [VMaMamba-ND: Selective State Space Modeling for Multi-Dimensional Data](https://arxiv.org/abs/2402.05892)

  This paper believe itself is the general-implementation of Mamba in vision facing multi-dimensionaal data
   
   Contributions:

   - Layer-level and Block-level to deal multi-dimensional data.
     - Layer-level: in one mamba block, there are sevel directions to scan
     - Block-level: one mamba block with one certain directions. there are several such mamba block to scan data  
   - In Block-level, try H+H-W+W-T+T-, [H+H-][W+W-][T+T-], [H+H-W+W-][T+T-]. Find the sequence of H+H-W+W-T+T- is best. The code of this is very difficult to understand.
   - use three scan-dactorization policies. actually, I did not understand this fully.

<img src="https://github.com/BaoBao0926/Paper_reading/blob/main/Image/1.Mamba/1.1%20VisionMamba/1.1.1%20Backbone_network/Mamba-ND.png" alt="Model" style="width: 600px; height: auto;"/>
    
   <br />

</details>

<details> 
   <summary>
   <b style="font-size: larger;">1.1.1.4 Local Mamba </b>          <!--   1.1.1.4  Local Mamba   -->
   </summary>   
    
   The Paper: [LocalMamba: Visual State Space Model with Windowed Selective Scan](https://arxiv.org/pdf/2403.09338)

  Based on Vim and VMamba as baseline model. better performance than Vim and VMamba
   
   Contributions:

   - propose to use Fig.1.c way to scan. a local window first then next. two size, 2*2 window and 7*7 window
   - in one mamba block, use four different scan directions, choosing from
     - vertical, vertical-filp, horizontial, horizontial-flip, 2*2 window, 7*7 window
     - inspired by DARTS, using a trainable network and a learnable factor, choose in one layer which four directions from 8 above are used
   - SCAAttm, spatial and channel attention modules: this module can enhance the itegration of diverse features and eliminate extraneous information shown in Fig.3.b. It seems to like a attention module.

<img src="https://github.com/BaoBao0926/Paper_reading/blob/main/Image/1.Mamba/1.1%20VisionMamba/1.1.1%20Backbone_network/LocalMamba.png" alt="Model" style="width: 600px; height: auto;"/>
    
   <br />

</details>



</details>      <!--    -----------------------------------------  1.1.1 Vision Mamba Backbone Network  -------------------------------------------------------  -->

   
</details>      <!--    -----------------------------------------  1.1 Vision Mamba   -------------------------------------------------------  -->

</details>      <!--    -----------------------------------------  1. Mamba   -------------------------------------------------------  -->
