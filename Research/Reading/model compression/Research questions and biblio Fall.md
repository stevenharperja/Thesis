# Sources:


### Literature reviews searching for LRF materials
- Advances and Challenges in Large Model Compression: A Survey https://dl-acm-org.ezproxy.library.ewu.edu/doi/10.1145/3675417.3675487
    - Links:
        - Bert https://huggingface.co/blog/bert-101
        - LRF:
            - Chen P, Yu H F, Dhillon I, et al. Drone: Data-aware low-rank compression for
    large nlp models[J]. Advances in neural information processing systems, 2021, 
    34: 29321-29334. In NeurIPS. https://proceedings.neurips.cc/paper/2021/file/f56de5ef149cf0aedcc8f4797031e229-Paper.pdf 
                - Conducts LRF on the dataset apparently.
                - Provides a nice explanation for the O() time stats of LRF as compression.
                - Only provides speedup?
            - Edalati A, Tahaei M, Rashid A, et al. Kronecker decomposition for gpt compression
                - Uses GLUE benchmark
                - https://math.stackexchange.com/questions/4190516/how-to-decompose-a-matrix-as-the-sum-of-kronecker-products 
    [J]. arXiv preprint arXiv:2110.08152, 2021
    - thoughts:
        - Doesn't it make sense anyways that you can randomly prune and models will still work? since a lot of people use dropout layers? How does the results of pruning change when its on layers which dropout and those which dont? do dropout layers work differently than I think? do they not save all of the nodes by default?
        - I should be looking up model compression surveys, then just looking only at the low rank factorization bits, that will let me know if the idea has been done.

///Nothing to do with LRF, I think? but has cool info.
- Model Compression in Practice: Lessons Learned from Practitioners Creating On-device Machine Learning Experiences https://dl-acm-org.ezproxy.library.ewu.edu/doi/10.1145/3613904.3642109
    - Dynamic models
        - Worth looking into, they seem rather involved? might need architecture-specific techniques? it sounds cool though. 
            - Early Exit
            - Gated
                - Reminds me of multimodal models, or chatgpt using wolfram.
    - They say [17 , 19, 22 , 68, 101] are good survey papers, that is:
        - Yu Cheng, Duo Wang, Pan Zhou, and Tao Zhang. 2018. Model compression and
            acceleration for deep neural networks: The principles, progress, and challenges.
            IEEE Signal Processing Magazine 35, 1 (2018), 126–136. https://doi.org/10.1109/
            msp.2017.2765695
        - Tejalal Choudhary, Vipul Mishra, Anurag Goswami, and Jagannathan Saranga-
            pani. 2020. A comprehensive survey on model compression and acceleration.
            Artifcial Intelligence Review 53, 7 (2020), 5113–5155. https://doi.org/10.1007/
            s10462-020-09816-7
        - Lei Deng, Guoqi Li, Song Han, Luping Shi, and Yuan Xie. 2020. Model compres-
            sion and hardware acceleration for neural networks: A comprehensive survey.
            Proc. IEEE 108, 4 (2020), 485–532. https://doi.org/10.1109/jproc.2020.2976475
        - Gaurav Menghani. 2023. Efcient deep learning: A survey on making deep
            learning models smaller, faster, and better. Comput. Surveys 55, 12 (2023), 1–37.
        - Marcos Treviso, Ji-Ung Lee, Tianchu Ji, Betty van Aken, Qingqing Cao, Manuel R
            Ciosici, Michael Hassid, Kenneth Heafeld, Sara Hooker, Colin Rafel, et al . 2023.
            Efcient methods for natural language processing: A survey. Transactions of
            the Association for Computational Linguistics 11 (2023), 826–860.
    - Palettization
        - using something like a lookup table can make a model smaller. but it still keeps the speed.
    - https://dovetail.com/research/inductive-coding/
        - Not really sure what this is. Its not software related.
    - Lots of good insights into what mindset or workflows to use when implementing model compression for a project.
    - Decision trees, SVM, and other such methods can be great for high efficiency on very low end devices.
    - “If you want to go to lower bit quantization, such as
        4 or below, it’s almost impossible to use post-training
        quantization because the difference in accuracy gets
        way too big. So for this level of compression you need
        to do training-aware compression.” — E9
    - F1 score might be a helpful indicator on some metrics? See 5.6

    ///Continue at 5.6.2 or maybe move on to something else with LRF
    


- Efficient Compressing and Tuning Methods for Large Language Models: A Systematic Literature Review https://dl-acm-org.ezproxy.library.ewu.edu/doi/10.1145/3728636
    - Has a LOT of techniques for LRF
    - https://arxiv.org/abs/2106.09685 LoRA
    - Fisher-weighted SVD 
        Yen-Chang Hsu, Ting Hua, Sung-En Chang, Qiang Lou, Yilin Shen, and Hongxia Jin. 2022. Language model compres-
        sion with weighted low-rank factorization. In Proceedings of the 10th International Conference on Learning Represen-
        tations, Virtual Event, April 25–29, 2022.
    - true-weighted SVD (TFWSVD)
        Ting Hua, Yen-Chang Hsu, Felicity Wang, Qiang Lou, Yilin Shen, and Hongxia Jin. 2022. Numerical optimizations
        for weighted low-rank estimation on language models. In Proceedings of the 2022 Conference on Empirical Methods
        in Natural Language Processing, Abu Dhabi, United Arab Emirates, December 7–11, 2022.
    - low-rank and sparse approximation (LoSparse)
        StarCoder: May the source be with
        you! Transactions on Machine Learning Research. 1–55
    - low-precision low-rank factorization (LPLR) 
        Rajarshi Saha, Varun Srivastava, and Mert Pilanci. 2023. Matrix compression via randomized low rank and low
        precision factorization. In Proceedings of the Advances in Neural Information Processing Systems 36: Annual Conference
        on Neural Information Processing Systems 2023, New Orleans, LA, USA, December 10–16, 2023
    - progressive low-rank decomposition  
        Habib Hajimolahoseini, Mehdi Rezagholizadeh, Vahid Partovinia, Marzieh S. Tahaei, Omar Mohamed Awad, and
        Yang Liu. 2021. Compressing pre-trained language models using progressive low rank decomposition. In Advances in
        Neural Information Processing Systems 35: Annual Conference on Neural Information Processing Systems 2021, December
        6–14, 2021, Virtual
    - low-rank decomposition (LoRD)
        Ayush Kaushal, Tejas Vaidhya, and Irina Rish. 2023. LORD: Low rank decomposition of monolingual code LLMs for
        one-shot compression. In Proceedings of the International Conference on Machine Learning, 23–29 July 2023, Honolulu,
        Hawaii, USA.
    - ESPACE
        Charbel Sakr and Brucek Khailany. 2024. ESPACE: Dimensionality reduction of activations for model compression.
        In Proceedings of the Advances in Neural Information Processing Systems 38: Annual Conference on Neural Information
        Processing Systems, Vancouver, BC, Canada, December 10–15, 2024.
    - Drone and Kroneger also mentioned.
    

- From Summer reading: https://arxiv.org/pdf/2308.07633 Survey, with info on benchmarking strategies.   A Survey on Model Compression for Large Language Models
    - Low-Rank Factorization 
        Nathan Srebro and Tommi S. Jaakkola. 2003.
        Weighted low-rank approximations. In Ma-
        chine Learning, Proceedings of the Twentieth
        International Conference (ICML 2003), August
        21-24, 2003, Washington, DC, USA, pages 720–
        727. AAAI Press.
    - LPLR also mentioned
    - ASVD 
        Zhihang Yuan, Yuzhang Shang, Yue Song, Qiang
        Wu, Yan Yan, and Guangyu Sun. 2023b.
        ASVD: activation-aware singular value decom-
        position for compressing large language mod-
        els. CoRR, abs/2312.05821.
    - LASER
        Pratyusha Sharma, Jordan T. Ash, and Dipendra
        Misra. 2024. The truth is in there: Improv-
        ing reasoning with layer-selective rank reduc-
        tion. In The Twelfth International Conference
        on Learning Representations
    



- Yu Cheng, Duo Wang, Pan Zhou, and Tao Zhang. 2018. Model compression and
    acceleration for deep neural networks: The principles, progress, and challenges.
    IEEE Signal Processing Magazine 35, 1 (2018), 126–136. https://doi.org/10.1109/
    msp.2017.2765695
    - canonical polyadic (CP) decomposition of kernel tensors
        V. Lebedev, Y. Ganin, M. Rakhuba, I. V. Oseledets, and V. S. Lempitsky,
        “Speeding-up convolutional neural networks using fine-tuned CP-decomposition,”
        Computing Res. Repository, vol. abs/1412.6553, 2014. [Online]. Available: https://
        arxiv.org/abs/1412.6553
    - fully connected classical
         M. Denil, B. Shakibi, L. Dinh, M. Ranzato, and N. D. Freitas. (2013).
        Predicting parameters in deep learning. Advances in Neural Information
        Processing Systems, 26, 2148–2156. [Online]. Available: http://media.nips.cc/nips-
        books/nipspapers/paper_files/nips26/1053.pdf
    - fully connected classif layers
        T. N. Sainath, B. Kingsbury, V. Sindhwani, E. Arisoy, and B. Ramabhadran,
        “Low-rank matrix factorization for deep neural network training with high-dimen-
        sional output targets,” in Proc. IEEE Int. Conf. Acoustics Speech Signal
        Processing, 2013, pp. 6655–6659


- Tejalal Choudhary, Vipul Mishra, Anurag Goswami, and Jagannathan Saranga-
    pani. 2020. A comprehensive survey on model compression and acceleration.
    Artifcial Intelligence Review 53, 7 (2020), 5113–5155. https://doi.org/10.1007/
    s10462-020-09816-7
    - Can't access.

- A Survey on Model Compression and Acceleration for Pretrained Language Models 
            Canwen Xu, Julian McAuley https://arxiv.org/abs/2202.07105
    - How much does low rank decomposition reduce time taken to multiply matrices?
    - Drone mentioned, and Kronecker
    - ALBERT (Lan et al. 2020) uses factorization for the embedding layer
        Lan, Z.; Chen, M.; Goodman, S.; et al. 2020. ALBERT: A
        Lite BERT for Self-supervised Learning of Language Repre-
        sentations. In ICLR.
    - self-attentive factorized embeddings (SAFE) 
        Reid, M.; Marrese-Taylor, E.; and Matsuo, Y. 2021. Sub-
        former: Exploring Weight Sharing for Parameter Efficiency
        in Generative Transformers. In EMNLP (Findings).
    - good info on benchmarks for NLP, and measurements 


- Lei Deng, Guoqi Li, Song Han, Luping Shi, and Yuan Xie. 2020. Model compres-
    sion and hardware acceleration for neural networks: A comprehensive survey.
    Proc. IEEE 108, 4 (2020), 485–532. https://doi.org/10.1109/jproc.2020.2976475
    -   Huynh and
        Won [82] proposed a new training method to achieve
        acceleration, which is compatible with the SVD format
        for networks with a single hidden layer. 
        -   H. T. Huynh and Y. Won, “Training single hidden
            layer feedforward neural networks by singular
            value decomposition,” in Proc. 4th Int. Conf.
            Comput. Sci. Converg. Inf. Technol., 2009
    -   Masana et al.
        [83] used SVD to decompose the product of the input
        and the weight matrix. 
        -   M. Masana, J. V. D. Weijer, L. Herranz,
            A. D. Bagdanov, and J. M. Alvarez,
            “Domain-adaptive deep network compression,” in
            Proc. IEEE Int. Conf. Comput. Vis. (ICCV),
            Oct. 2017, pp. 4289–4297
    -   In the scenario of
        distributed training, Yu et al. [91] utilized principal compo-
        nent analysis (PCA) to linearly project the weight gradients
        into a low-dimensional space that enables fast decentral-
        ized gradient aggregation (e.g., ring all-reduce) in the
        compressed domain
        -   M. Yu et al., “GradiVeQ: Vector quantization for
            bandwidth-efficient gradient aggregation in
            distributed CNN training,” in Proc. Adv. Neural Inf.
            Process. Syst., 2018, pp. 5125–5135
    -   there are also some
        other matrix decomposition methods such as QR and
        CUR 
        -   N. Kishore Kumar and J. Schneider, “Literature
            survey on low rank approximation of matrices,”
            Linear Multilinear Algebra, vol. 65, no. 11,
            pp. 2212–2244, Dec. 2016
    -   a variant similar to CUR called Nyström method
        -   A. Gittens and M. W. Mahoney, “Revisiting the
            Nyström method for improved large-scale
            machine learning,” J. Mach. Learn. Res., vol. 17,
            no. 1, pp. 3977–4041, 2016.
    -   "In a nutshell, SVD is the best matrix decomposition
        method to compress neural networks with overall high quality"
        - so dont look into any of those ig?
    -   The paper also mentions lots of methods for tensor decomposition. not sure if I need that rn though.
    -   Mostly this paper gives techniques for decomposition it sounds like.
    - Shim et al. [90] utilized SVD to compress the last softmax layer
        for large vocabulary neural networks
        -   K. Shim, M. Lee, I. Choi, Y. Boo, and W. Sung,
            “SVD-softmax: Fast softmax approximation on
            large vocabulary neural networks,” in Proc. Adv.
            Neural Inf. Process. Syst., 2017, pp. 5463–5473.
    - Includes Tensor Decomposition # 
    - ![alt text](images/tensors%20and%20matrixes.png)
    -   Here, we first introduce an operation between a tensor
        and a matrix called mode-i contracted product
        -   N. Lee and A. Cichocki, “Fundamental tensor
            operations for large-scale data analysis using
            tensor network formats,” Multidimensional Syst.
            Signal Process., vol. 29, no. 3, pp. 921–960,
            Mar. 2017.
            -   Renamed to: Fundamental Tensor Operations for Large-Scale Data Analysis in Tensor Train Formats
    - Tucker Decomposition and classical prolongation (CP) Decomposition
        -   T. G. Kolda and B. W. Bader, “Tensor
            decompositions and applications,” SIAM Rev.,
            vol. 51, no. 3, pp. 455–500, Aug. 2009
        -   " Hence, CP is typically used for
            interpreting data components while Tucker is often used
            for data compression"
    - https://rumn.medium.com/convex-vs-non-convex-functions-why-it-matters-in-optimization-for-machine-learning-39cd9427dfcc
    -   Instead of directly replacing weights with decomposed tensors,
        Chien and Bao [101] considered the entire decomposi-
        tion process of (19) as a connection, similar to [102]
        and [103]. Specifically, the input tensor X is treated as
        the input layer, the kernel tensor K is treated as the
        output layer, and the weights are replaced with sequential
        mode-i contracted products among pseudo-inverse fac-
        tor matrices of F (i), then the model accuracy would be
        improved
        -   J. Kossaifi, Z. C. Lipton, A. Khanna, T. Furlanello,
            and A. Anandkumar, “Tensor regression
            networks,” 2017, arXiv:1707.08308. [Online].
            Available: http://arxiv.org/abs/1707.08308
    - Higher Order SVD / Multilinear SVD
        -   L. De Lathauwer, B. De Moor, and J. Vandewalle,
            “A multilinear singular value decomposition,”
            SIAM J. Matrix Anal. Appl., vol. 21, no. 4,
            pp. 1253–1278, 2000.
    -   Matrix Orthogonality?
        -   https://en.wikipedia.org/wiki/Orthogonal_matrix
    -   Tensor overview
        -   https://web.stanford.edu/class/ee270/scribes/lecture4.pdf
    -   hierarchical tucker (HT) and tensor train (TT)
        -   "Although the HT format has flexible organization forms
            (because the count of different dimension trees with a
            fixed number of nodes is distributed as Catalan num-
            bers [125]), it suffers from an intractable obstacle to obtain
            an optimal specific form. As a result, just a few practices of
            HT can be found [126]–[128]. Among them, only Cohen
            and Shashua [128] and Cohen et al. [129] showed neural
            network applications, but they are yet to answer how to
            obtain an optimal HT form."
            -   [128] N. Cohen and A. Shashua, “Convolutional rectifier
                networks as generalized tensor decompositions,”
                in Proc. 33rd Int. Conf. Mach. Learn., vol. 48,
                2016, pp. 955–963
            -   [129] N. Cohen, O. Sharir, and A. Shashua, “On the
                expressive power of deep learning: A tensor
                analysis,” J. Mach. Learn. Res., Workshop Conf.
                Proc., vol. 49, pp. 1–31, Jun. 2016
    -   "or Tucker and CP, the combination of them especially
        BTD [107], [112] presents promising potential in
        neural network compression."
        -   [107] Y. Chen, X. Jin, B. Kang, J. Feng, and S. Yan,
            “Sharing residual units through collective tensor
            factorization to improve deep neural networks,” in
            Proc. 27th Int. Joint Conf. Artif. Intell., Jul. 2018,
            pp. 635–641.
        -   [112] J. Ye et al., “Learning compact recurrent neural
            networks with block-term tensor decomposition,”
            in Proc. IEEE/CVF Conf. Comput. Vis. Pattern
            Recognit., Jun. 2018, pp. 9378–9387.
    -    The TC format [136]
        (also called tensor ring [118]) is a variant of TT,
        -   [118] Q. Zhao, M. Sugiyama, L. Yuan, and A. Cichocki,
            “Learning efficient tensor representations with
            ring-structured networks,” in Proc. ICASSP IEEE
            Int. Conf. Acoust., Speech Signal Process. (ICASSP),
            May 2019, pp. 8608–8612.
        -   [136] M. Espig, K. K. Naraparaju, and J. Schneider,
            “A note on tensor chain approximation,” Comput.
            Vis. Sci., vol. 15, no. 6, pp. 331–344, 2012
        -   [137] Q. Zhao, G. Zhou, S. Xie, L. Zhang, and
            A. Cichocki, “Tensor ring decomposition,” 2016,
            arXiv:1606.05535. [Online]. Available:
            http://arxiv.org/abs/1606.05535
        

- Tensor Decomposition 
    -   https://www.youtube.com/playlist?list=PL-dpUY5nCWZhOlc8XM398ps9A0CEDNGIH
        -   Tamara Talk https://www.youtube.com/watch?v=L8uT6hgMt00&pp=ygUUdGVuc29yIGRlY29tcG9zaXRpb24%3D
            -   Matrix Ideas that have been apparently extended to tensors
                -   Independent Component Analysis? ICA
                -   Nonnegative Matrix Factorization
                -   Sparse Matrix Factorization
                -   Matrix Completion
            -   Rank 1 matrices are created by the outer product of two vectors. Thats it. its real simple. Then the sum of rank 1 matrices to do a low rank approx can be then written as a series of outer products. so its just a sum of each term of a[i]*b[j] for each a,b being the inputs to the outer products to make the ij'th term of all the rank 1 matrices. 6:38 and before explains this.
            -   CP has many names and was invented thrice
                - Canonical Polyadic is the nowadays accepted name
                - CANDECOMP Canonical Decomposition
                - Parallel Factors PARAFAC
            -   Computing CP 
                -   NP-Hard, even impossible for small tensors apparently, approximate instead.
                -   Getting a rank 2 from a rank 1 isn't a sub problem, you have to recompute the whole thing? (Kolda 2001)
                -   Factor matrices are not orthogonal, and also may have lin. dep. columns (18:59)
                -   Factor matrices are unique solution ish.
                -   Can solve using Least squares, fixing the values of the other matrices while you focus on a current matrix. (20:10)
                    - nonconvex problem, but the subproblems for each matrix are convex.
                    - using khatri-rao product
                    - very overdetermined system, use randomization with CPRAND algorithm to solve this. randomized least squares
                    - Dont actually store full matrix/tensor as those forms, just pull out pieces
                        - Memory is a larger concern than speed here.
                    - Johnson-Lindenstrauss transforms fixes some properties of the resulting matrix?? (24:28)
                    - To check for convergence, take a poll of a random subset of terms to see if converged under accuracy we want
                        - Chernoff-Hoeffding theory says how many samples are good number to take.
                    -   Instead of Least squares, theres also other CP Decomp methods
                        -   Better for data which is NOT continuous gaussian.
                    -   Rayleigh CP(31:42)
                        - good for non-negative data
                        - Rayleigh is a distribution thats a little bit like normal distrib but positive. (kinda reminds me of the values that that imagenet model had)
                    -   Boolean CP
                        - for discrete (noncontinuous) data. Eg. 0s and 1s.
                        - bernoulli distribtuion is relevant
                    -   Poisson for count data
                    -    Can use many types of objective functions for what kind of data you're working with. (35:35)
                    -   Gradient of this has MTTKRP form ???
                        - Type of kernel?
                    -   graph data makes a sparse binary dataset (38:39)
                        -    can use boolean CP
                -   Knowing how the data is distributed is important to what techniques you apply and how you analyze it. In my case, my data is tensor/matrix values within a pretrained neural net.
                -   Tensortoolbox for matlab
                -   Parallel CP and GCP implementations: gitlab.com/tensors/genten
                -   https://kolda.net/
                    - Has links to her software, including a tucker thing?
                    - "C:\Users\shwes\Documents\school\Thesis\Research\Reading\model compression\books\Tensor Textbook Final Draft Mar 25 2025.pdf"


- https://kolda.net/post/tensor-textbook-final-draft/
    "C:\Users\shwes\Documents\school\Thesis\Research\Reading\model compression\books\Tensor Textbook Final Draft Mar 25 2025.pdf"
    -   nonnegative matrix factorization 


                    
    -   https://www.youtube.com/watch?v=ooY4qjomlPo&pp=ygUUdGVuc29yIGRlY29tcG9zaXRpb24%3D CP



- Gaurav Menghani. 2023. Efcient deep learning: A survey on making deep
    learning models smaller, faster, and better. Comput. Surveys 55, 12 (2023), 1–37.
    - not much stuff on compression here.



- Marcos Treviso, Ji-Ung Lee, Tianchu Ji, Betty van Aken, Qingqing Cao, Manuel R
    Ciosici, Michael Hassid, Kenneth Heafeld, Sara Hooker, Colin Rafel, et al . 2023.
    Efcient methods for natural language processing: A survey. Transactions of
    the Association for Computational Linguistics 11 (2023), 826–860.
    - Not much here either




#### Thoughts:

It could be useful to look at pruning techniques in addition to Low rank approx. since they are basically the same thing, just with a change of basis.

### Reading LRF papers

- Chen P, Yu H F, Dhillon I, et al. Drone: Data-aware low-rank compression for
large nlp models[J]. Advances in neural information processing systems, 2021, 
34: 29321-29334. In NeurIPS. https://proceedings.neurips.cc/paper/2021/file/f56de5ef149cf0aedcc8f4797031e229-Paper.pdf 
    - Conducts LRF on the dataset apparently.
    - Provides a nice explanation for the O() time stats of LRF as compression.
    - Only provides speedup?
- Edalati A, Tahaei M, Rashid A, et al. Kronecker decomposition for gpt compression
    - Uses GLUE benchmark
    - https://math.stackexchange.com/questions/4190516/how-to-decompose-a-matrix-as-the-sum-of-kronecker-products 

- low-precision low-rank factorization (LPLR) 
    Rajarshi Saha, Varun Srivastava, and Mert Pilanci. 2023. Matrix compression via randomized low rank and low
    precision factorization. In Proceedings of the Advances in Neural Information Processing Systems 36: Annual Conference
    on Neural Information Processing Systems 2023, New Orleans, LA, USA, December 10–16, 2023
    - https://en.wikipedia.org/wiki/Johnson%E2%80%93Lindenstrauss_lemma
    - https://en.wikipedia.org/wiki/Dither
    - They use LLAMA.
    - Other model quant compression works:  
        -   T. Dettmers, M. Lewis, Y. Belkada, and L. Zettlemoyer. LLM.int8 (): 8-bit matrix multiplication
            for transformers at scale. arXiv preprint arXiv:2208.07339, 2022. (Cited on pages 7 and 9)
        -   E. Frantar, S. Ashkboos, T. Hoefler, and D. Alistarh. Gptq: Accurate post-training quantization
            for generative pre-trained transformers. arXiv preprint arXiv:2210.17323, 2022. (Cited on
            page 9
        -   E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, and W. Chen. LoRA:
            Low-rank adaptation of large language models. In International Conference on Learning
            Representations, 2022. URL https://openreview.net/forum?id=nZeVKeeFYf9. (Cited
            on pages 1 and 9)
    - They don't give any benchmarks for model accuracy. However, Frobenius norm error is a good thing to track layer by layer it seems
    - Sharding a model
        - Mohammad Shoeybi, Mostofa Patwary, Raul Puri, Patrick LeGresley, Jared Casper, and Bryan
            Catanzaro. Megatron-lm: Training multi-billion parameter language models using model par-
            allelism, 2020.
    - They use GLUE for BERT models
    - For GPT-2 they use a setup similar to:
        -   Xiang Lisa Li and Percy Liang. Prefix-Tuning: Optimizing Continuous Prompts for Generation.
            arXiv:2101.00190 [cs], January 2021. URL http://arxiv.org/abs/2101.00190.
    - GPU is NVIDIA Tesla V100
        - Not sure how this compares to what we have but its likely wayyyyy faster.
        - Maybe I can look up papers on how to run models on edge devices to get stuff which could reliably run and be trained on our stuff.


- LORA: LOW-RANK ADAPTATION OF LARGE LANGUAGE MODELS https://arxiv.org/abs/2106.09685 
    - Looks hella useful. AND hella relevant for me to read all the way through.
    - Uses RoBERTa, DeBERTa, GPT-2, and GPT-3 models
    - Their technique is to only LRA the changes made when fine-tuning off a pre-trained model. This allows them also to
        hot-swap these changes since they are stored seperately
    - Later I think I wanna dig into their code. Or maybe build off of it.

- Fisher-weighted SVD 
    Yen-Chang Hsu, Ting Hua, Sung-En Chang, Qiang Lou, Yilin Shen, and Hongxia Jin. 2022. Language model compres-
    sion with weighted low-rank factorization. In Proceedings of the 10th International Conference on Learning Represen-
    tations, Virtual Event, April 25–29, 2022.
    - Factors in the significance of each term into how it should be modified during SVD, like how pruning does.
    - Fisher Information
        - Razvan Pascanu and Yoshua Bengio. Revisiting natural gradient for deep networks. In In Interna-
            tional Conference on Learning Representations (ICLR), 2014
        - A Tutorial on Fisher Information  https://arxiv.org/pdf/1705.01064
    - Could be combined with LoRA into one thing? or do LoRA and then do FWSVD on the whole network including the frozen weights? How does LoRA work anyway?
    - Uses lots of benchmarks
    

//TOREAD
- true-weighted SVD (TFWSVD)
    Ting Hua, Yen-Chang Hsu, Felicity Wang, Qiang Lou, Yilin Shen, and Hongxia Jin. 2022. Numerical optimizations
    for weighted low-rank estimation on language models. In Proceedings of the 2022 Conference on Empirical Methods
    in Natural Language Processing, Abu Dhabi, United Arab Emirates, December 7–11, 2022.
- low-rank and sparse approximation (LoSparse)
    StarCoder: May the source be with
    you! Transactions on Machine Learning Research. 1–55
- progressive low-rank decomposition  
    Habib Hajimolahoseini, Mehdi Rezagholizadeh, Vahid Partovinia, Marzieh S. Tahaei, Omar Mohamed Awad, and
    Yang Liu. 2021. Compressing pre-trained language models using progressive low rank decomposition. In Advances in
    Neural Information Processing Systems 35: Annual Conference on Neural Information Processing Systems 2021, December
    6–14, 2021, Virtual
- low-rank decomposition (LoRD)
    Ayush Kaushal, Tejas Vaidhya, and Irina Rish. 2023. LORD: Low rank decomposition of monolingual code LLMs for
    one-shot compression. In Proceedings of the International Conference on Machine Learning, 23–29 July 2023, Honolulu,
    Hawaii, USA.
- ESPACE
    Charbel Sakr and Brucek Khailany. 2024. ESPACE: Dimensionality reduction of activations for model compression.
    In Proceedings of the Advances in Neural Information Processing Systems 38: Annual Conference on Neural Information
    Processing Systems, Vancouver, BC, Canada, December 10–15, 2024.
- Low-Rank Factorization 
    Nathan Srebro and Tommi S. Jaakkola. 2003.
    Weighted low-rank approximations. In Ma-
    chine Learning, Proceedings of the Twentieth
    International Conference (ICML 2003), August
    21-24, 2003, Washington, DC, USA, pages 720–
    727. AAAI Press.
- ASVD 
    Zhihang Yuan, Yuzhang Shang, Yue Song, Qiang
    Wu, Yan Yan, and Guangyu Sun. 2023b.
    ASVD: activation-aware singular value decom-
    position for compressing large language mod-
    els. CoRR, abs/2312.05821.
- LASER
    Pratyusha Sharma, Jordan T. Ash, and Dipendra
    Misra. 2024. The truth is in there: Improv-
    ing reasoning with layer-selective rank reduc-
    tion. In The Twelfth International Conference
    on Learning Representations
- ALBERT (Lan et al. 2020) uses factorization for the embedding layer
    Lan, Z.; Chen, M.; Goodman, S.; et al. 2020. ALBERT: A
    Lite BERT for Self-supervised Learning of Language Repre-
    sentations. In ICLR.
- self-attentive factorized embeddings (SAFE) 
    Reid, M.; Marrese-Taylor, E.; and Matsuo, Y. 2021. Sub-
    former: Exploring Weight Sharing for Parameter Efficiency
    in Generative Transformers. In EMNLP (Findings).
- canonical polyadic (CP) decomposition of kernel tensors
    V. Lebedev, Y. Ganin, M. Rakhuba, I. V. Oseledets, and V. S. Lempitsky,
    “Speeding-up convolutional neural networks using fine-tuned CP-decomposition,”
    Computing Res. Repository, vol. abs/1412.6553, 2014. [Online]. Available: https://
    arxiv.org/abs/1412.6553
- fully connected classical
        M. Denil, B. Shakibi, L. Dinh, M. Ranzato, and N. D. Freitas. (2013).
    Predicting parameters in deep learning. Advances in Neural Information
    Processing Systems, 26, 2148–2156. [Online]. Available: http://media.nips.cc/nips-
    books/nipspapers/paper_files/nips26/1053.pdf
- fully connected classif layers
    T. N. Sainath, B. Kingsbury, V. Sindhwani, E. Arisoy, and B. Ramabhadran,
    “Low-rank matrix factorization for deep neural network training with high-dimen-
    sional output targets,” in Proc. IEEE Int. Conf. Acoustics Speech Signal
    Processing, 2013, pp. 6655–6659
-   In the scenario of
    distributed training, Yu et al. [91] utilized principal compo-
    nent analysis (PCA) to linearly project the weight gradients
    into a low-dimensional space that enables fast decentral-
    ized gradient aggregation (e.g., ring all-reduce) in the
    compressed domain
    -   M. Yu et al., “GradiVeQ: Vector quantization for
        bandwidth-efficient gradient aggregation in
        distributed CNN training,” in Proc. Adv. Neural Inf.
        Process. Syst., 2018, pp. 5125–5135
-   Huynh and
        Won [82] proposed a new training method to achieve
        acceleration, which is compatible with the SVD format
        for networks with a single hidden layer. 
        -   H. T. Huynh and Y. Won, “Training single hidden
            layer feedforward neural networks by singular
            value decomposition,” in Proc. 4th Int. Conf.
            Comput. Sci. Converg. Inf. Technol., 2009
-   Masana et al.
    [83] used SVD to decompose the product of the input
    and the weight matrix. 
    -   M. Masana, J. V. D. Weijer, L. Herranz,
        A. D. Bagdanov, and J. M. Alvarez,
        “Domain-adaptive deep network compression,” in
        Proc. IEEE Int. Conf. Comput. Vis. (ICCV),
        Oct. 2017, pp. 4289–4297
-   In the scenario of
    distributed training, Yu et al. [91] utilized principal compo-
    nent analysis (PCA) to linearly project the weight gradients
    into a low-dimensional space that enables fast decentral-
    ized gradient aggregation (e.g., ring all-reduce) in the
    compressed domain
    -   M. Yu et al., “GradiVeQ: Vector quantization for
        bandwidth-efficient gradient aggregation in
        distributed CNN training,” in Proc. Adv. Neural Inf.
        Process. Syst., 2018, pp. 5125–5135
-   there are also some
    other matrix decomposition methods such as QR and
    CUR 
    -   N. Kishore Kumar and J. Schneider, “Literature
        survey on low rank approximation of matrices,”
        Linear Multilinear Algebra, vol. 65, no. 11,
        pp. 2212–2244, Dec. 2016
-   a variant similar to CUR called Nyström method
    -   A. Gittens and M. W. Mahoney, “Revisiting the
        Nyström method for improved large-scale
        machine learning,” J. Mach. Learn. Res., vol. 17,
        no. 1, pp. 3977–4041, 2016.
- Shim et al. [90] utilized SVD to compress the last softmax layer
    for large vocabulary neural networks
    -   K. Shim, M. Lee, I. Choi, Y. Boo, and W. Sung,
        “SVD-softmax: Fast softmax approximation on
        large vocabulary neural networks,” in Proc. Adv.
        Neural Inf. Process. Syst., 2017, pp. 5463–5473.