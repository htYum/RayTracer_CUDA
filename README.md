# RayTracer_CUDA
using cuda with ***GTX 1066***

>因为对CUDA的使用和gpu架构的不熟悉，以及前期参考了前一本mini book的CUDA代码导致这本next week相关代码也采用了想当然的结构  
结果就是显卡占用率几乎为0，混乱的显存管理（在显卡里搞递归和BVH不可取），以及超长的处理时间（虽然比我cpu还是快，但这速度真的不敢恭维）  
目前仍在学习相关知识，希望之后能有足够的能力在这之上优化或者重写，完成效果更好的渲染器
#### 最终场景  
没有做原书中的柏林噪声贴图  
删减了地板的个数  
以及去掉了那个1000个小球的立方体（巨耗时间，因为BVH的创建和递归查找有大问题，想用CUDA搞BVH的同学建议参考优秀论文，不过大概也不会有人写出我这种稀烂的代码）  
分辨率800*800 采样数5000 渲染时间6.5h（太丢人了，不过好在5000采样的效果还行，用下一本书的蒙特卡洛方法应该会大幅提升效果  
![main](https://github.com/htYum/RayTracer_CUDA/blob/ch3_final_scene/x64/Release/main.png)

--------  

#### 之前测试效果用的cornell box  
分辨率1920*1080 采样数8000 渲染时间14h（丢人  
![1](https://github.com/htYum/RayTracer_CUDA/blob/ch3_final_scene/x64/Release/1.png)
