## Xception

[Xception paper](<https://arxiv.org/abs/1610.02357>)

![è¿éåå¾çæè¿°](https://img-blog.csdn.net/20170715083812048?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvdTAxNDM4MDE2NQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

<https://blog.csdn.net/u014380165/article/details/75142710>

![è¿éåå¾çæè¿°](https://img-blog.csdn.net/20170715083646403?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvdTAxNDM4MDE2NQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

`depthwise separable convolution`  two steps

- channel-wise  spatial conv   the fig. (b)
- 1x1 conv the fig. (c)

Xception作为Inception v3的改进，主要是在Inception v3的基础上引入了depthwise separable convolution，在基本不增加网络复杂度的前提下提高了模型的效果。有些人会好奇为什么引入depthwise separable convolution没有大大降低网络的复杂度，因为depthwise separable convolution在mobileNet中主要就是为了降低网络的复杂度而设计的。原因是作者加宽了网络，使得参数数量和Inception v3差不多，然后在这前提下比较性能。因此Xception目的不在于模型压缩，而是提高性能。

[作者：AI之路 
来源：CSDN 
原文：https://blog.csdn.net/u014380165/article/details/75142710 
版权声明：本文为博主原创文章，转载请附上博文链接！]()



## Aligned Xception



![Snipaste_2019-06-07_17-20-36](C:\Users\liu-z\Desktop\Snipaste_2019-06-07_17-20-36.png)

More recently, the MSRA team [31] modifies
the Xception model (called Aligned Xception) and further **pushes the performance in the task of object detection**