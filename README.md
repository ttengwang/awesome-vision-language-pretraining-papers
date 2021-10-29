# Recent Advances in Vision and Language PreTrained Models (VL-PTMs)
I reorganize the list in chronological order and add some new papers.

The original repo is maintained by [WANG Yue](https://yuewang-cuhk.github.io/) (wangyue2714@gmail.com). Last update on 2021/06/14.


*Update log:**

 2021-10-29: add CVPR'21, ICCV'21, ACL'21, ICML'21 papers


## Table of Contents


 [Image-based VL-PTMs](#image-based-vl-ptms)
  * [Representation Learning](#representation-learning)
  * [Task-specific](#task-specific)
  * [Other Analysis](#other-analysis)

   [Video-based VL-PTMs](#video-based-vl-ptms)

   [Speech-based VL-PTMs](#speech-based-vl-ptms)

   [Other Transformer-based multimodal networks](#other-transformer-based-multimodal-networks)

   [Other Resources](#other-resources)


# Image-based VL-PTMs

## Representation Learning
### Arxiv
[2021.08] [SimVLM: Simple Visual Language Model Pretraining with Weak Supervision](https://arxiv.org/pdf/2108.10904.pdf)
<font color="red"> `SOTA on VQA` </font>

[2021.05] [SemVLP: Vision-Language Pre-training by Aligning Semantics at Multiple Levels](https://openreview.net/forum?id=Wg2PSpLZiH)

[2021.05] [Multi-modal Understanding and Generation for Medical Images and Text via Vision-Language Pre-Training](https://arxiv.org/abs/2105.11333)  
<font color="red"> `Medical image precessing` </font>

[2020.12] [LAMP: Label Augmented Multimodal Pretraining](https://arxiv.org/pdf/2012.04446.pdf)

[2020.12]  [A Closer Look at the Robustness of Vision-and-Language Pre-trained Models](https://arxiv.org/abs/2012.08673), 
<font color="red"> `In-depth Analysis` </font>

[2020.11] [Multimodal Pretraining Unmasked: Unifying the Vision and Language BERTs](https://arxiv.org/pdf/2011.15124.pdf)

[2020.10] [CAPT: Contrastive Pre-Training for Learning Denoised Sequence Representations](https://arxiv.org/pdf/2010.06351.pdf), 

[2020.07] [How Much Can CLIP Benefit Vision-and-Language Tasks?](https://arxiv.org/abs/2107.06383) [ICLR 2021 submission](https://openreview.net/forum?id=zf_Ll3HZWgy)

[2020.04] [Pixel-BERT: Aligning Image Pixels with Text by Deep Multi-Modal Transformers](https://arxiv.org/abs/2004.00849), 

[2020.03] [InterBERT: Vision-and-Language Interaction for Multi-modal Pretraining](https://arxiv.org/abs/2003.13198)


### 2021


[ICML]  [ViLT: Vision-and-Language Transformer Without Convolution or Region Supervision](https://arxiv.org/pdf/2102.03334.pdf) [[code]](https://github.com/dandelin/ViLT)

[ICML]  [Unifying Vision-and-Language Tasks via Text Generation](https://arxiv.org/abs/2102.02779)  
<font color="red"> `Multi-task Learning` </font>

[ICML] [Scaling Up Visual and Vision-Language Representation Learning With Noisy Text Supervision](https://arxiv.org/abs/2102.05918)  
<font color="red"> `Dataset perspective` </font>

[NIPS] [Align before Fuse: Vision and Language Representation Learning with Momentum Distillation](https://arxiv.org/abs/2107.07651) [[code]](https://github.com/salesforce/ALBEF/)

[ICCV] [COOKIE: Contrastive Cross-Modal Knowledge Sharing Pre-training for Vision-Language Representation]( https://openaccess.thecvf.com/content/ICCV2021/papers/Wen_COOKIE_Contrastive_Cross-Modal_Knowledge_Sharing_Pre-Training_for_Vision-Language_Representation_ICCV_2021_paper.pdf) [[code]](https://github.com/kywen1119/COOKIE)

[ICCV] [Airbert: In-Domain Pretraining for Vision-and-Language Navigation](https://arxiv.org/abs/2108.09105) [[code]](https://github.com/airbert-vln/airbert)

[CVPR] [Seeing Out of the Box: End-to-End Pre-Training for Vision-Language Representation Learning](https://arxiv.org/abs/2104.03135)

[CVPR] [UC2: Universal Cross-Lingual Cross-Modal Vision-and-Language Pre-Training](https://arxiv.org/abs/2104.00332)

[CVPR] [Conceptual 12M: Pushing Web-Scale Image-Text Pre-Training To Recognize Long-Tail Visual Concepts](https://arxiv.org/abs/2102.08981)  
<font color="red"> `Dataset` </font>

[CVPR] [Multimodal Contrastive Training for Visual Representation Learning](https://arxiv.org/abs/2104.12836)

[CVPR] [Kaleido-BERT: Vision-Language Pre-Training on Fashion Domain](https://arxiv.org/abs/2103.16110)

[CVPR] [Causal Attention for Vision-Language Tasks](https://arxiv.org/abs/2103.03493) [[code]](https://github.com/yangxuntu/catt)

[CVPR] [M3P: Learning Universal Representations via Multitask Multilingual Multimodal Pre-Training](https://arxiv.org/abs/2006.02635) [[code]](https://github.com/microsoft/M3P)

[CVPR] [VinVL: Revisiting Visual Representations in Vision-Language Models](https://arxiv.org/abs/2101.00529) [[detection code](https://github.com/microsoft/Oscar)] [[pretraining code](https://github.com/microsoft/Oscar)]

[ACL] [UNIMO: Towards Unified-Modal Understanding and Generation via Cross-Modal Contrastive Learning](https://arxiv.org/abs/2012.15409) [[code]](https://github.com/PaddlePaddle/Research/tree/master/NLP/UNIMO)

[ACL] [E2E-VLP: End-to-End Vision-Language Pre-training Enhanced by Visual Learning](https://aclanthology.org/2021.acl-long.42.pdf)

[AAAI] [Scheduled Sampling in Vision-Language Pretraining with Decoupled Encoder-Decoder Network](https://arxiv.org/pdf/2101.11562.pdf)

[AAAI] [ERNIE-ViL: Knowledge Enhanced Vision-Language Representations Through Scene Graph](https://arxiv.org/abs/2006.16934)

### 2020


[NIPS] [Large-Scale Adversarial Training for Vision-and-Language Representation Learning](https://arxiv.org/abs/2006.06195)
<font color="red"> `Adversarial Training` </font>

[ICLR] [VL-BERT: Pre-training of Generic Visual-Linguistic Representations](https://arxiv.org/abs/1908.08530) [[code]](https://github.com/jackroos/VL-BERT)

[CVPR]  [12-in-1: Multi-Task Vision and Language Representation Learning](https://arxiv.org/abs/1912.02315) [[code]](https://github.com/facebookresearch/vilbert-multi-task)  
<font color="red"> `Multi-task Learning` </font>

[ACL] [VisualBERT: A Simple and Performant Baseline for Vision and Language](https://arxiv.org/abs/1908.03557)[[code]](https://github.com/uclanlp/visualbert)

[AAAI] [Unicoder-VL: A Universal Encoder for Vision and Language by Cross-modal Pre-training](https://arxiv.org/abs/1908.06066)

[AAAI] [Unified Vision-Language Pre-Training for Image Captioning and VQA](https://arxiv.org/pdf/1909.11059.pdf) [[code]](https://github.com/LuoweiZhou/VLP) (**VLP**)

[ECCV] [Behind the Scene: Revealing the Secrets of Pre-trained Vision-and-Language Models](https://arxiv.org/abs/2005.07310)
<font color="red"> `In-depth Analysis` </font>

[ECCV] [UNITER: Learning Universal Image-text Representations](https://arxiv.org/abs/1909.11740), ECCV 2020, [[code]](https://github.com/ChenRocks/UNITER)

[ECAI] [Weak Supervision helps Emergence of Word-Object Alignment and improves Vision-Language Tasks](https://arxiv.org/abs/1912.03063)

[ECCV] [Oscar: Object-Semantics Aligned Pre-training for Vision-Language Tasks](https://arxiv.org/pdf/2004.06165.pdf) [[code]](https://github.com/microsoft/Oscar)

[ACMMM] [DeVLBert: Learning Deconfounded Visio-Linguistic Representations](https://arxiv.org/abs/2008.06884) [[code]](https://github.com/shengyuzhang/DeVLBert)

[EMNLP] [X-LXMERT: Paint, Caption and Answer Questions with Multi-Modal Transformers](https://arxiv.org/abs/2009.11278) [[code]](https://github.com/allenai/x-lxmert)

### 2019

[NIPS] [ViLBERT: Pretraining Task-Agnostic Visiolinguistic Representations for Vision-and-Language Tasks](https://arxiv.org/abs/1908.02265) [[code]](https://github.com/jiasenlu/vilbert_beta)

[EMNLP] [LXMERT: Learning Cross-Modality Encoder Representations from Transformers](https://arxiv.org/abs/1908.07490) [[code]](https://github.com/airsplay/lxmert)

## Task-specific

### Arxiv 

[ICLR submission] [Cross-Probe BERT for Efficient and Effective Cross-Modal Search](https://openreview.net/forum?id=bW9SYKHcZiz), ICLR 2021 submission.

[2020.02] [What BERT Sees: Cross-Modal Transfer for Visual Question Generation](https://arxiv.org/abs/2002.10832)  
<font color="red"> `Visual Question Generation` </font>

[2020.01] [ImageBERT: Cross-Modal Pre-training with Large-scale Weak-supervised Image-text Data](https://arxiv.org/abs/2001.07966)  
<font color="red"> `Text-image retrieval` </font>


### 2021


[CVPR] [A Recurrent Vision-and-Language BERT for Navigation](https://arxiv.org/abs/2011.13922)  
<font color="red"> `Vision-language navigation` </font>

[AAAI] [VisualMRC: Machine Reading Comprehension on Document Images](https://arxiv.org/abs/2101.11272), (**LayoutT5, LayoutBART**)
<font color="red"> `Machine Reading Comprehension` </font>

[IEEE Access] [Visual Relationship Detection With Visual-Linguistic Knowledge From Multimodal Representations](https://ieeexplore.ieee.org/document/9387302)    
<font color="red"> `Visual Relationship Detection` </font>

[NLPCC] [XGPT: Cross-modal Generative Pre-Training for Image Captioning](https://arxiv.org/abs/2003.01473)  
<font color="red"> `Image captioning` </font>


### 2020 


[CVPR] [Iterative Answer Prediction with Pointer-Augmented Multimodal Transformers for TextVQA](https://arxiv.org/abs/1911.06258) [[code]](https://github.com/ronghanghu/pythia/tree/project/m4c/projects/M4C), (**M4C**)  
<font color="red"> `TextVQA` </font>

[CVPR] [Towards Learning a Generic Agent for Vision-and-Language Navigation via Pre-training](https://arxiv.org/abs/2002.10638) [[code]](https://github.com/weituo12321/PREVALENT), (**PREVALENT**)  
<font color="red"> `Vision-Language Navigation` </font>

[ECCV] [Large-scale Pretraining for Visual Dialog: A Simple State-of-the-Art Baseline](https://arxiv.org/abs/1912.02379) [[code]](https://github.com/vmurahari3/visdial-bert), (**VisDial-BERT**)  
<font color="red"> `Visual Dialog` </font>

[EMNLP] [VD-BERT: A Unified Vision and Dialog Transformer with BERT](https://arxiv.org/abs/2004.13278), EMNLP 2020 [[code]](https://github.com/salesforce/VD-BERT), (**VD-BERT**)  
<font color="red"> `Visual Dialog` </font>

[EMNLP] [STL-CQA: Structure-based Transformers with Localization and Encoding for Chart Question Answering](https://www.aclweb.org/anthology/2020.emnlp-main.264.pdf), EMNLP 2020.  
<font color="red"> `Chart VQA` </font>

### 2019

[EMNLP] [Fusion of Detected Objects in Text for Visual Question Answering](https://arxiv.org/abs/1908.05054) [[code]](https://github.com/google-research/language/tree/master/language/question_answering/b2t2), (**B2T2**)  
<font color="red"> `Visual Question Answering` </font>

## Other Analysis

[ICCVW 2021] [Are we pretraining it right? Digging deeper into visio-linguistic pretraining](https://arxiv.org/abs/2004.08744),  
<font color="red"> `In-depth Analysis` </font>

[ACL SRW 2020] [Adaptive Transformers for Learning Multimodal Representations](https://arxiv.org/abs/2005.07486)  
<font color="red"> `Adaptive Analysis` </font>

[arXiv 2020.04] [Deep Multimodal Neural Architecture Search](https://arxiv.org/abs/2004.12070)  
<font color="red"> `Neural Architecture Search` </font>

[arXiv 2020.02] [Measuring Social Biases in Grounded Vision and Language Embeddings](https://arxiv.org/abs/2002.08911) [[code]](https://github.com/candacelax/bias-in-vision-and-language)  
<font color="red"> `Social Bias in VL Embedding` </font>



# Video-based VL-PTMs
### Arxiv


[2020.07] [Auto-captions on GIF: A Large-scale Video-sentence Dataset for Vision-language Pre-training](https://arxiv.org/abs/2007.02375)

[2020.02]  [UniVL: A Unified Video and Language Pre-Training Model for Multimodal Understanding and Generation](https://arxiv.org/abs/2002.06353) [[code]](https://github.com/microsoft/UniVL)

[2020.01]  [Learning Spatiotemporal Features via Video and Text Pair Discrimination](https://arxiv.org/abs/2001.05691) [[code]](https://github.com/MCG-NJU/CPD-Video) (**CPD**)

[2019.08]  [M-BERT: Injecting Multimodal Information in the BERT Structure](https://arxiv.org/abs/1908.05787)

[2019.06]  [Learning Video Representations Using Contrastive Bidirectional Transformers](https://arxiv.org/abs/1906.05743) [[ICLR 2020 submission]](https://openreview.net/forum?id=rJgRMkrtDr) (**CBT**)

### 2021


[ACL Findings] [VLM: Task-agnostic Video-Language Model Pre-training for Video Understanding](https://arxiv.org/abs/2105.09996)

[CVPR]  [Less is More: CLIPBERT for Video-and-Language Learning via Sparse Sampling](https://arxiv.org/pdf/2102.06183.pdf)

[ICLR]  [Parameter Efficient Multimodal Transformers for Video Representation Learning](https://arxiv.org/pdf/2012.04124.pdf)

### 2020

[CVPR] [ActBERT: Learning Global-Local Video-Text Representations](http://openaccess.thecvf.com/content_CVPR_2020/html/Zhu_ActBERT_Learning_Global-Local_Video-Text_Representations_CVPR_2020_paper.html)

[EMNLP]  [HERO: Hierarchical Encoder for Video+Language Omni-representation Pre-training](https://arxiv.org/abs/2005.00200)

[ACL]  [Video-Grounded Dialogues with Pretrained Generation Language Models](https://arxiv.org/abs/2006.15319)

[AACL]  [Multimodal Pretraining for Dense Video Captioning](https://arxiv.org/pdf/2011.11760.pdf)

[AAAIW]  [Bridging Text and Video: A Universal Multimodal Transformer for Video-Audio Scene-Aware Dialog](https://arxiv.org/abs/2002.00163)

### 2019

[ICCV] [VideoBERT: A Joint Model for Video and Language Representation Learning](https://arxiv.org/abs/1904.01766)

[ICCVW] [BERT for Large-scale Video Segment Classification with Test-time Augmentation](https://arxiv.org/abs/1912.01127) [[code]](https://github.com/hughshaoqz/3rd-Youtube8M-TM)

# Speech-based VL-PTMs
[arXiv 2019.11]  [Effectiveness of self-supervised pre-training for speech recognition](https://arxiv.org/abs/1911.03912)

[arXiv 2019.10]  [SpeechBERT: Cross-Modal Pre-trained Language Model for End-to-end Spoken Question Answering](https://arxiv.org/abs/1910.11559)

[arXiv 2019.10]  [Vq-wav2vec: Self-Supervised Learning of Discrete Speech Representations](https://arxiv.org/abs/1910.05453)

[arXiv 2019.06] [Towards Transfer Learning for End-to-End Speech Synthesis from Deep Pre-Trained Language Models](https://arxiv.org/abs/1906.07307)

[arXiv 2019.09]  [Understanding Semantics from Speech Through Pre-training](https://arxiv.org/abs/1909.10924)

# Other Resources

 Two recent surveys on pretrained language models
  * [Pre-trained Models for Natural Language Processing: A Survey](https://arxiv.org/abs/2003.08271), arXiv 2020/03
  * [A Survey on Contextual Embeddings](https://arxiv.org/abs/2003.07278), arXiv 2020/03

   Other surveys about multimodal research
  * [Trends in Integration of Vision and Language Research: A Survey of Tasks, Datasets, and Methods](https://doi.org/10.1613/jair.1.11688), JAIR 2021
  * [Deep Multimodal Representation Learning: A Survey](https://ieeexplore.ieee.org/abstract/document/8715409), arXiv 2019 
  * [Multimodal Machine Learning: A Survey and Taxonomy](https://arxiv.org/abs/1705.09406), TPAMI 2018
  * [A Comprehensive Survey of Deep Learning for Image Captioning](https://arxiv.org/abs/1810.04020), ACM Computing Surveys 2018

   Other repositories of relevant reading list
  * [Pre-trained Languge Model Papers from THU-NLP](https://github.com/thunlp/PLMpapers)
  * [BERT-related Papers](https://github.com/tomohideshibata/BERT-related-papers)
  * [Reading List for Topics in Multimodal Machine Learning](https://github.com/pliang279/awesome-multimodal-ml)
  * [A repository of vision and language papers](https://github.com/sangminwoo/awesome-vision-and-language-papers)

