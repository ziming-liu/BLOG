## refineDet



### motivation

`one stage`目标检测准确率低的原因是因为 `class imbalance problem`。

已有的一些解决方案｛

- object prior constraint[24]
- reshaping the standard cross entropy loss to focus on hard examples[28]
- max-out labeling mechanism[53]

｝

`two stage` 工作的advantages｛

- two stage 结构做 sampling heuristics 采样探索 来处理 class imbalance problem
- 两步级联地去回归 object box parameters
- two stage features 来描述objects--> RPN 特征 预测二分类（object / background）， stage2 预测 多分类（BG and objects classes）  **本文借鉴了这种两次分类的好处**

｝



### structure 两个模块

- anchor refinement module ARM *****
- objection detection module  ODM

![Snipaste_2019-05-13_16-18-43.png](https://github.com/ziming-liu/BLOG/blob/master/object%20detection/refineDet/src/Snipaste_2019-05-13_16-18-43.png?raw=true)

整个模型的结构可以分解为三部分 

- backbone                                                                                ---------> Features[F1,F2,F3,F4]
- necks  包括  FPN（blue path ） +  ARM（red path）       --------->refine anchors[A1,...A4] +FPNfeatures[F1'....F4'] 
- head (SSD head 或者retina head)                                       ----------->predict multi classes + bbox loc

为了方便理解，我们可以把ARM模块化成红色的path， 把FPN 特征金字塔化成蓝色的path。这样理解， `refineDet`的`anchor refine` 就比较好理解成一个module了.

ARM 实际上很像two stage 模型里RPN 承担的作用，paper中也说是在`imitate` two stage模型



### 细节

- 采用`jaccard overlap`[7] 对应anchors和GTbox， 1  match给每个gtbox 一个与它overlap最大的anchor， 2 match所有anchor boxs 到 与其overlap 大于0.5 的gtbox。

- anchor 选择的正负样本比例  nagtives:positives==3:1  ，选择 loss值大的 负样本anchors，而不是选择全部负样本anchor或者随机选择负样本anchor



### 代码

<https://github.com/luuuyi/RefineDet.PyTorch>

模型代码

```python
#main model
class RefineDet(nn.Module):
    """Single Shot Multibox Architecture
    The network is composed of a base VGG network followed by the
    added multibox conv layers.  Each multibox layer branches into
        1) conv2d for class conf scores
        2) conv2d for localization predictions
        3) associated priorbox layer to produce default bounding
           boxes specific to the layer's feature map size.
    See: https://arxiv.org/pdf/1512.02325.pdf for more details.

    Args:
        phase: (string) Can be "test" or "train"
        size: input image size
        base: VGG16 layers for input, size of either 300 or 500
        extras: extra layers that feed to multibox loc and conf layers
        head: "multibox head" consists of loc and conf conv layers
    """

    def __init__(self, phase, size, base, extras, ARM, ODM, TCB, num_classes):
        super(RefineDet, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        self.cfg = (coco_refinedet, voc_refinedet)[num_classes == 21]
        self.priorbox = PriorBox(self.cfg[str(size)])
        with torch.no_grad():
            self.priors = self.priorbox.forward()
        self.size = size

        # SSD network
        self.vgg = nn.ModuleList(base)
        # Layer learns to scale the l2 normalized features from conv4_3
        self.conv4_3_L2Norm = L2Norm(512, 10)
        self.conv5_3_L2Norm = L2Norm(512, 8)
        self.extras = nn.ModuleList(extras)

        self.arm_loc = nn.ModuleList(ARM[0])
        self.arm_conf = nn.ModuleList(ARM[1])
        self.odm_loc = nn.ModuleList(ODM[0])
        self.odm_conf = nn.ModuleList(ODM[1])
        #self.tcb = nn.ModuleList(TCB)
        self.tcb0 = nn.ModuleList(TCB[0])
        self.tcb1 = nn.ModuleList(TCB[1])
        self.tcb2 = nn.ModuleList(TCB[2])

        if phase == 'test':
            self.softmax = nn.Softmax(dim=-1)
            self.detect = Detect_RefineDet(num_classes, self.size, 0, 1000, 0.01, 0.45, 0.01, 500)

    def forward(self, x):
        """Applies network layers and ops on input image(s) x.

        Args:
            x: input image or batch of images. Shape: [batch,3,300,300].

        Return:
            Depending on phase:
            test:
                Variable(tensor) of output class label predictions,
                confidence score, and corresponding location predictions for
                each object detected. Shape: [batch,topk,7]

            train:
                list of concat outputs from:
                    1: confidence layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]
                    3: priorbox layers, Shape: [2,num_priors*4]
        """
        sources = list()
        tcb_source = list()
        arm_loc = list()
        arm_conf = list()
        odm_loc = list()
        odm_conf = list()
--------------------1------------------------------------------------
        # apply vgg up to conv4_3 relu and conv5_3 relu
        for k in range(30):
            x = self.vgg[k](x)
            if 22 == k:
                s = self.conv4_3_L2Norm(x)
                sources.append(s)
            elif 29 == k:
                s = self.conv5_3_L2Norm(x)
                sources.append(s)

        # apply vgg up to fc7
        for k in range(30, len(self.vgg)):
            x = self.vgg[k](x)
        sources.append(x)

        # apply extra layers and cache source layer outputs
        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:
                sources.append(x)
-----------------------------------------------------------------
------------------------2-----------------------------------------------
        # apply ARM and ODM to source layers
        for (x, l, c) in zip(sources, self.arm_loc, self.arm_conf):
            arm_loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            arm_conf.append(c(x).permute(0, 2, 3, 1).contiguous())
        arm_loc = torch.cat([o.view(o.size(0), -1) for o in arm_loc], 1)
        arm_conf = torch.cat([o.view(o.size(0), -1) for o in arm_conf], 1)
        #print([x.size() for x in sources])
        # calculate TCB features
        #print([x.size() for x in sources])
----------------------------------------------------------------------
-----------------------3----------------------------------------------
        p = None
        for k, v in enumerate(sources[::-1]):
            s = v
            for i in range(3):
                s = self.tcb0[(3-k)*3 + i](s)
                #print(s.size())
            if k != 0:
                u = p
                u = self.tcb1[3-k](u)
                s += u
            for i in range(3):
                s = self.tcb2[(3-k)*3 + i](s)
            p = s
            tcb_source.append(s)
        #print([x.size() for x in tcb_source])
        tcb_source.reverse()
-------------------------------------------------------------------------
------------------------4---------------------------------------------
        # apply ODM to source layers
        for (x, l, c) in zip(tcb_source, self.odm_loc, self.odm_conf):
            odm_loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            odm_conf.append(c(x).permute(0, 2, 3, 1).contiguous())
        odm_loc = torch.cat([o.view(o.size(0), -1) for o in odm_loc], 1)
        odm_conf = torch.cat([o.view(o.size(0), -1) for o in odm_conf], 1)
        #print(arm_loc.size(), arm_conf.size(), odm_loc.size(), odm_conf.size())
-------------------------------------------------------------------------
------------------------------5-------------------------------------
        if self.phase == "test":
            #print(loc, conf)
            output = self.detect(
                arm_loc.view(arm_loc.size(0), -1, 4),           # arm loc preds
                self.softmax(arm_conf.view(arm_conf.size(0), -1,
                             2)),                               # arm conf preds
                odm_loc.view(odm_loc.size(0), -1, 4),           # odm loc preds
                self.softmax(odm_conf.view(odm_conf.size(0), -1,
                             self.num_classes)),                # odm conf preds
                self.priors.type(type(x.data))                  # default boxes
            )
        else:
            output = (
                arm_loc.view(arm_loc.size(0), -1, 4),
                arm_conf.view(arm_conf.size(0), -1, 2),
                odm_loc.view(odm_loc.size(0), -1, 4),
                odm_conf.view(odm_conf.size(0), -1, self.num_classes),
                self.priors
            )
        return output

```

主要是forward 方法，简单分成五个part分析。

1- backbone卷积网络提取特征，得到输出应该是四个尺度的features [F1,..F4]

2- ARM模块预测得到bbox的回归和anchor正负样本的二分类结果。，这个结果暂存，arm_loc + arm_conf

3- paper中的TCB 模块从高层特征向底层特征融合，实际上类似FPN特征金字塔，得到新的四个尺度features [F1'...F4']

4- 第三步的四个尺度特征送入 anchor head ，预测bbox的回归和所有object多类别的分类结果。odm_loc + odm_conf.

5- 如果是test时，把arm_loc + arm_conf    odm_loc + odm_conf. 送入一个detect的类，预测得到最终的bbox。  

解释一下这个过程，在目标检测中我们预测的bbox实际上是一个编码了的长度为4的vector,而不是实际的坐标或者长宽（这一点如果不知道看faster RCNN等paper）。

那么， 在第五步，我们送入detect类完成的就是根据预测的bbox location解码（decode）得到实际的坐标。

首先， anchor refine module(ARM)预测得到arm_loc,  

```python
arm_anchors = decode(arm_Loc,predefined_anchor)
#predefined_anchor是参数定义的anchors，它没有编码，就是实际坐标值，也就是faster RCNN中说的default  anchors
#arm_loc 是ARM模块预测的、编码的、预测loc
#arm_anchors = 根据 上面两个参数解码出来预测bbox的实际空间坐标值。
```

其次， ODM模块预测得到的odm_loc，**就不是根据事先定义的default anchors去解码了，而是根据刚刚得到的`arm_anchors` 去解码。**

**这也就是本文的核心-----------> `anchor refine`**

```python
odm_bbox = decode(odm_loc, arm_anchors)
# odm_loc是预测的、编码了的 bbox loc
# arm_anchors and odm_bbox都是实际的空间坐标。
```



上面是test阶段，如果是train时，就不需要refine这个anchor啦， 直接把

output = (
                arm_loc.view(arm_loc.size(0), -1, 4),
                arm_conf.view(arm_conf.size(0), -1, 2),
                odm_loc.view(odm_loc.size(0), -1, 4),
                odm_conf.view(odm_conf.size(0), -1, self.num_classes),
                self.priors
            )

送出去计算loss咯， 跟two stage 的模型一样，总的loss 有四项，` arm_Loc arm_conf   odm_loc odm_conf`

其他代码细节，可以参考 https://github.com/luuuyi/RefineDet.PyTorch

