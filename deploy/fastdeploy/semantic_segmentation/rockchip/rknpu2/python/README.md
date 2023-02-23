[English](README.md) | 简体中文
# PaddleSeg RKNPU2 Python部署示例

本目录下用于展示PaddleSeg系列模型在RKNPU2上的部署，以下的部署过程以PPHumanSeg为例子。

## 1. 部署环境准备 
在部署前，需确认以下步骤

- 1. 软硬件环境满足要求，RKNPU2环境部署等参考[FastDeploy环境要求](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/cn/faq/rknpu2/rknpu2.md)

## 2. 部署模型准备

模型转换代码请参考[模型转换文档](../README.md)

## 3. 运行部署示例  

本目录下提供`infer.py`快速完成PPHumanseg在RKNPU上部署的示例。执行如下脚本即可完成

```bash
# 下载部署示例代码
git clone https://github.com/PaddlePaddle/PaddleSeg.git 
cd PaddleSeg/deploy/fastdeploy/semantic_segmentation/rockchip/rknpu2/python

# 下载图片
wget https://paddleseg.bj.bcebos.com/dygraph/pp_humanseg_v2/images.zip
unzip images.zip

# 运行部署示例
python3 infer.py --model_file ./Portrait_PP_HumanSegV2_Lite_256x144_infer/Portrait_PP_HumanSegV2_Lite_256x144_infer_rk3588.rknn \
                --config_file ./Portrait_PP_HumanSegV2_Lite_256x144_infer/deploy.yaml \
                --image images/portrait_heng.jpg
```

## 4. 注意事项
RKNPU上对模型的输入要求是使用NHWC格式，且图片归一化操作会在转RKNN模型时，内嵌到模型中，因此我们在使用FastDeploy部署时，需要先调用DisableNormalizeAndPermute(C++)或`disable_normalize_and_permute(Python)，在预处理阶段禁用归一化以及数据格式的转换。

## 5. 更多指南

- [FastDeploy部署PaddleSeg模型概览](..)
- [PaddleSeg RKNPU2 C++部署](../cpp)
- [转换PaddleSeg模型至RKNN模型文档](../README.md)

## 6. 常见问题
- [如何将模型预测结果SegmentationResult转为numpy格式](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/cn/faq/vision_result_related_problems.md)