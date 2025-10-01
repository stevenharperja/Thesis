•	Wouldn’t it be cool to apply a ML model on the weights of multiple different models trained for the same task to create new models with different weights which produce similar results or smth? Kinda too hard though and I can’t think of a use case.
o	I want to look into “pruning” models and techniques for that.
•	https://pytorch.org/tutorials/intermediate/pruning_tutorial.html
•		https://arxiv.org/abs/1803.03635 Sounds super cool.
•	https://www.datature.io/blog/a-comprehensive-guide-to-neural-network-model-pruning 
oLook up “model compression”?

What sounds interesting to me:
-	Quantization-Aware Training
o	Bitnet 
	https://github.com/microsoft/BitNet 
	https://arxiv.org/pdf/2402.17764 
o	
-	Model Distillation
o	Self-distillation
-	One-Shot Pruning
o	https://proceedings.mlr.press/v202/frantar23a.html SparseGPT (referenced in SAMSP and FLAP etc.)
o	https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10445737 SAMSP
o	https://ojs.aaai.org/index.php/AAAI/article/view/28960 FLAP
o	https://openreview.net/forum?id=PxoFut3dWW Wanda (I’ve heard of this one before I feel like)
o	https://openreview.net/forum?id=vXxardq6db SliceGPT (PCA pruning) seems cool and I think itd be interpretable, though maybe specific to Transformers arch. 
o	https://doi.org/10.18653/V1/2020.EMNLP-MAIN.608 Dynamic pruning
-	Low Rank Factorization
o	How does it change the architecture?
o	Existing architectures in the llm paper:
	https://aaai.org/papers/ICML03-094-weighted-low-rank-approximations/ 
	https://papers.nips.cc/paper_files/paper/2023/hash/3bf4b55960aaa23553cd2a6bdc6e1b57-Abstract-Conference.html 
	https://arxiv.org/abs/2312.05821
	https://openreview.net/forum?id=ozX92bu8VA 
-	AutoML


Questions:
-	If you pruned the weights of a matrix, couldn’t you reduce the error induced by removing weights by: adjusting the remaining weights in some direction based on the like “dot product” that that matrix gives? And attempting to make the dot product results closer to how they were based on the original inputs. Or perhaps not even the dot product directly, but some sort of inner product or latent results within the model?
o	Maybe that’s how the algorithms which have sensitivity metrics based on second derivatives work?
-	If soft relu is used in a model, you could probably make it slightly simpler by replacing it with regular relu at runtime. Miniscule speed improvement though I think.
-	Why not add a regularization parameter to loss functions (a constraint to the model loss) which makes it so that the model should prioritize having low rank matrices? Then you could just make them low rank to start, but if they cant be low rank then they become high rank. Would it make training slower? Could it be done on an existing model? Would the architecture need to be changed? How does low rank factorization usually change the architecture?
o	Low rank approximations are approximating an optimal set of weights that were found. But I feel like surely there must be a common possibility where there exists a set of  weights which are low rank but also are fairly optimal, if not equivalently optimal to weights trained in high rank. Maybe even with lower rank than what could be approximated to a given set of weights while preserving those weights accuracy? Especially because the LASER paper appatently found improvements in some cases of low rank approxing.
o	Using the sum aggregate of the spectral values, that is, the trace of AtA, is just the L1 regularization. Because trace of AtA is the sum of each weight param squared. Then the derivative of it takes out the square. So it’s the same. So we need a way to target just the higher order spectral values instead of just the trace.
	https://en.m.wikipedia.org/wiki/Pseudo-determinant 
	https://en.m.wikipedia.org/wiki/Spectrum_of_a_matrix 
	https://mathoverflow.net/questions/462766/is-there-a-fast-way-to-check-if-a-matrix-has-any-small-eigenvalues 
	https://math.stackexchange.com/questions/1796742/easiest-way-to-compute-singular-values-of-matrix#1796841 
	https://en.m.wikipedia.org/wiki/Spectral_radius What if I subtracted the spectral radius from the trace? Maybe that would help a little?
	https://builtin.com/articles/svd-algorithm has code
	Would (AtA)^2 have singular values which favor the lower order ones as being bigger? Could I compare the difference between the traces for this matrix and  (AtA)^1 as a metric to measure the higher order singular values? I could test this or prove and see what this does to the singular values. Maybe I could ask my math teacher if hed be willing to check my work after.
•	Trace (AtA^2)/trace(AtA) might be good? Its like a power 4 frobenius norm over a 2 frobenius norm, just without the sqrts and stuff. I wanna see how this goes?
•	When I tried it with a 4x4 matrix of 1,2,…16, the score improved when I converted that matrix to low rank approximation with 1 singular value. I didn’t try it with the others though
•	To prevent it from favoring large numbers in general, maybe trace((AtA)^2) /(trace(AtA))^2 would be better? Or maybe the numerator should be sqrtd? Something like that
•	It seems to work alright. Orthonormal matrices are apparently very spread in their singular value stuffs.  So its likely the case that any time that a matrix can be rank reduced, it doesn’t rotate much, and just scales? Maybe. Anyways, adding the square onto that seems to counteract any scaling action pretty well. See the octave code in the /exploring/lowrank stuff. 
•	The bounds seem to be [1,0] so maybe using it as a multiplier would work better than adding a number to the loss function. Maybe 1/the value though so it maximizes it. Maybe cap it at 1000 or something if possible though. Actually maybe just use (1 – the value) so we don’t have to deal with numerical instability and infinity and whatnot.
o	Actually do 2-the value, because we don’t want the model to cheat by making it go to zero.
o	Actually maybe don’t use it as a multiplier, itd force the model to get gains by making bad matrixes. So maybe lets do it some other way instead. Maybe with addition and a lambda parameter yea. Maybe ask Bojian what he thinks about it.
	According to the linalg book section 10.4, the frobenius norm is the sum of the squares of the singular values. That could be a good one to use? Maybe we want to maximize the frobenius norm while minimizing the absolute values of the sum of each individual weight. Basically e want just a few of the singular values to be very large, while having the rest be quite small
•	I think instead of minimizing the sum of the weights, I ant to minimize the sum of the singular values. How can I do that? 
o	The Trace norm does this apparently? Or the trace norm measures the rank of the matrix? https://mathoverflow.net/questions/278013/what-is-the-intuition-for-the-trace-norm-nuclear-norm 
	Find Bojian’s slides on how to calculate the regularization parameter as a gradient descent minimization thing.
	The ideal with using this whole regularization parameter thing is to favor creating matrixes which have almost lower rank (values close to a row rank matrix). So that matrixes can be converted to low rank matrixes easier and with better accuracy. Low rank matrixes have few relevant singular values. Many of the higher order ones are zero or close to it. We want to create matrixes with as many higher order singular values which are close to zero as is practical, as measured by gradient descent on the error function.
	https://mathformachines.com/posts/eigenvalues-and-singular-values/ 
	https://mathoverflow.net/questions/278013/what-is-the-intuition-for-the-trace-norm-nuclear-norm trace norm
	https://homepages.cwi.nl/~fehr/QIT2021/Chap7.pdf more on trace norm
o	How can I test if my thing helps?
	I can run the MNIST test with the param, and without the param. Then on both models, run low rank approximation on all the matrices, and see the result at each rank reduction. Compare the difference. Test this with models created from the param set with a lambda scalar of 0.1,0.5,1,2,3,10,etc.
	Use Pytorch so that it is more approachable  for people to run and understand the code.
	Need to test differences using the same starting weights, and using multiple samples of starting weights
o	One thing to keep in mind is that in order for low rank approx. to be useful, the storage has to be less than the original storage matrix size. And hopefully the execution of multiplication should be less too? Or very limitedly higher?
o	Overall my idea is:
	Since low rank approximation has been shown to be effective, then there exists lower rank matrices which can have a low error
	Low rank approximation approximates: a high rank matrix with a low error for the dataset.
	Perhaps there exist low rank matrices with low error, that are lower than: the low rank approxies of the high rank matrixes. 
	To find this, we could use a regularization parameter to stay close to the “hyperplane” which contains low rank matrixes. 
	Then we can do low rank to essentially “project” onto that hyperplane without losing much information, since the numbers wont change much.
-	Would sharing weights between layers in an LLM be viable? How could that work in an attention layer? Has it been done before? I feel like someone would have. That CNN paper apparently did it a while ago. https://openreview.net/forum?id=rJgYxn09Fm 
realizations
-	Low rank factorizatin is just pruning but with a matrix transformation, or more linear algebra. And also recombining with a bit of extra steps.
-	Knowledge distillilation is just re-training a new model using fuzzy vector labels instead of one-hot vector labels.
 
Bibliography
To read:
https://medium.com/@zzxiang/6-kinds-of-model-compression-techniques-e0ae24bdd201 General overview
-	Cites: https://xailient.com/blog/4-popular-model-compression-techniques-explained/ 
o	“There are many more compression approaches beyond the four common ones covered in this article, such as weight sharing-based model compression, structural matrix, transferred filters, and compact filters.”
-	https://arxiv.org/abs/1710.09282 
https://arxiv.org/pdf/2308.07633 Survey, with info on benchmarking strategies.   A Survey on Model Compression for Large Language Models
•	https://huggingface.co/docs/transformers/perplexity Perplexity
o	https://thegradient.pub/understanding-evaluation-metrics-for-language-models/ 
o	https://medium.com/nlplanet/two-minutes-nlp-perplexity-explained-with-simple-probabilities-6cdc46884584 
	https://en.wikipedia.org/wiki/Entropy_(information_theory) 
•	Self- Distillation
o	https://ieeexplore.ieee.org/document/9381661 
•	https://arxiv.org/abs/2402.10631 Bitdistiller
•	https://arxiv.org/abs/2206.09557 quantized mat-mul	
•	https://arxiv.org/abs/2306.00978  AWQ 
o	https://github.com/mit-han-lab/llm-awq code
	https://github.com/mit-han-lab/llm-awq/tree/main/tinychat Tool for quantizing llms with examples
	https://www.ibm.com/think/topics/vision-language-models VLM
•	
•	https://arxiv.org/abs/2306.07629 squeeze llm
•	https://en.wikipedia.org/wiki/Vector_quantization Vector-wise quantization
o	https://en.wikipedia.org/wiki/Self-organizing_map random interesting 
	https://en.wikipedia.org/wiki/Elastic_map 
	https://en.wikipedia.org/wiki/Nonlinear_dimensionality_reduction#Principal_curves_and_manifolds 
•	https://huggingface.co/blog/not-lain/kv-caching What is kv cache
•	https://arxiv.org/abs/2402.02750 KIVI
•	https://proceedings.mlr.press/v202/frantar23a.html  Sparse GPT
o	Sparse regression
	https://stats.stackexchange.com/questions/110819/what-is-sparse-regression-model#158757 
•	https://en.wikipedia.org/wiki/Lasso_(statistics) 
o	https://www.statisticshowto.com/lasso-regression/ 
	https://cims.nyu.edu/~cfgranda/pages/OBDA_spring16/material/sparse_regression.pdf 
•	See section 2. The initial motivation of sparse regression is as a technique to reduce overfitting on small data sizes.
•	https://openreview.net/forum?id=PxoFut3dWW Wanda
•	https://ieeexplore.ieee.org/document/10445737 SAMSP, Hessian based
o	https://www.algebrapracticeproblems.com/hessian-matrix/ 
	https://en.wikipedia.org/wiki/Symmetry_of_second_derivatives 
	https://www.cuemath.com/algebra/symmetric-matrix/ 
•	https://ocw.mit.edu/courses/18-06sc-linear-algebra-fall-2011/804ab1e53134741d2044d241b50a285e_MIT18_06SCF11_Ses3.1sum.pdf 
•	dl.acm.org/doi/10.14778/3626292.3626303 Flash LLM
o	https://www.vldb.org/pvldb/vol17/p211-xia.pdf paper itself
o	https://cs184.eecs.berkeley.edu/public/sp19/lectures/lec-23-how-gpus-work/lec-23-how-gpus-work.pdf 
•	https://openreview.net/forum?id=J8Ajf9WfXP LLM-Pruner
o	https://medium.com/data-science-in-your-pocket/lora-for-fine-tuning-llms-explained-with-codes-and-example-62a7ac5a3578 LoRA (Low Rank Approx applied on the weight updates matrix)
•	https://ojs.aaai.org/index.php/AAAI/article/view/28960 FLAP
o	https://openreview.net/forum?id=PxoFut3dWW Wanda
•	https://openreview.net/forum?id=vXxardq6db Slice GPT (Uses PCA, how had this not been done before??)
o	https://mittaltushant.github.io/readings/intro.pdf Related? idk
•	https://openreview.net/forum?id=09iOdaeOzp Sheared Llama
o	https://tutorial.math.lamar.edu/classes/calciii/lagrangemultipliers.aspx Lagrange multiplier
•	https://doi.org/10.18653/V1/2020.EMNLP-MAIN.608 Dynamic pruning
•	SCOTT: Self-Consistent Chain-of-Thought Distillation https://aclanthology.org/2023.acl-long.304/ 
o	https://arxiv.org/abs/2210.15097 contrastive decoding
•	https://openreview.net/forum?id=1ndDmZdT4g Dynamic Sparse No Training ?? How does this work?
•	Knowledge distillation
o	https://www.ibm.com/think/topics/knowledge-distillation
o	I think its just making more prelabelled data using a large model, then training a small model on that data. Thereby getting more data to train a small model. That’s all it is??? It sounds like a hell of a lot of compute. Though I suppose it’d be faster at run time.
o	I guess using the teacher model also provides data for latent vectors to train on or intermediate word outputs in the case of autoregressive models. Which can be helpful? Something like that?
o	https://en.wikipedia.org/wiki/Knowledge_distillation 
o	Or maybe eg for classification, the teacher’s vector outputs can be used as the label, instead of a one-hot vector. That’s probably a more efficient way to train than using a new set of data. So I guess it would be faster to teach a smaller model? I think this is it! Probably.
o	Not sure I really wanna do knowledge distillation for my thesis. And it seems a little difficult to do.
•	Low Rank Factorization
o	LASER paper
	aa
•	AutoML (parameter learning)
o	https://proceedings.mlr.press/v70/finn17a.html Meta Learning
o	https://openreview.net/forum?id=r1Ue8Hcxg Neural Architecture Search
•	Learning Implicitly Recurrent CNNs Through Parameter Sharing https://openreview.net/forum?id=rJgYxn09Fm
https://arxiv.org/abs/2207.00112 Adjusting SVD to do low rank factorization
https://arxiv.org/abs/1710.09282 Survey, pretty old but could be good.
https://arxiv.org/pdf/2402.17764 Bitnet (quantization)
•	https://arxiv.org/abs/2310.11453 Bitnet
•	https://arxiv.org/abs/2104.09864 Rotary embeddings
•	https://mett29.github.io/posts/kv-cache/ kv cache
•	https://github.com/microsoft/BitNet?tab=readme-ov-file Same authors //TODO open this and run it now that conda is installed on my system. Its gotta be hella cool.
Quantization-Aware Training
•	https://www.tensorflow.org/model_optimization/guide/quantization/training 


Weighted Low-Rank Approximations https://aaai.org/papers/ICML03-094-weighted-low-rank-approximations/ 
•	From lin alg book: The singular value spectrum itself provides natural indicators for ap-proximation quality: 1. Sharp drops suggest clear truncation points, as with our temperature data where physical modes separate cleanly 2. Gradual decay without clear gaps warns that no natural low-rank approximation may exist 3. Clusters of similar singular values hint at coupled features requiring joint preservation
•	Also from linalg book 12.3, they go over low rank updates to matrixes which could be useful for training.
•	12.4 is also interesting
https://ocw.mit.edu/courses/18-409-algorithmic-aspects-of-machine-learning-spring-2015/a2b2f446a289a5a027bf67efaa54b7be_MIT18_409S15_chapp7.pdf low rank approx using random. And also more about the trace (nuclear) norm

LASER paper https://openreview.net/forum?id=ozX92bu8VA 
Randomized LRF and Low precision (quantization) https://papers.nips.cc/paper_files/paper/2023/hash/3bf4b55960aaa23553cd2a6bdc6e1b57-Abstract-Conference.html 
