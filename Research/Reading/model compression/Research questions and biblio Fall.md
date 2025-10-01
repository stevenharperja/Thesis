# Sources:

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
    






TOREAD 
- Efficient Compressing and Tuning Methods for Large Language Models: A Systematic Literature Review https://dl-acm-org.ezproxy.library.ewu.edu/doi/10.1145/3728636
TOREAD 
- Yu Cheng, Duo Wang, Pan Zhou, and Tao Zhang. 2018. Model compression and
    acceleration for deep neural networks: The principles, progress, and challenges.
    IEEE Signal Processing Magazine 35, 1 (2018), 126–136. https://doi.org/10.1109/
    msp.2017.2765695
TOREAD 
- Tejalal Choudhary, Vipul Mishra, Anurag Goswami, and Jagannathan Saranga-
    pani. 2020. A comprehensive survey on model compression and acceleration.
    Artifcial Intelligence Review 53, 7 (2020), 5113–5155. https://doi.org/10.1007/
    s10462-020-09816-7
TOREAD 
- Lei Deng, Guoqi Li, Song Han, Luping Shi, and Yuan Xie. 2020. Model compres-
    sion and hardware acceleration for neural networks: A comprehensive survey.
    Proc. IEEE 108, 4 (2020), 485–532. https://doi.org/10.1109/jproc.2020.2976475
TOREAD 
- Gaurav Menghani. 2023. Efcient deep learning: A survey on making deep
    learning models smaller, faster, and better. Comput. Surveys 55, 12 (2023), 1–37.
TOREAD 
- Marcos Treviso, Ji-Ung Lee, Tianchu Ji, Betty van Aken, Qingqing Cao, Manuel R
    Ciosici, Michael Hassid, Kenneth Heafeld, Sara Hooker, Colin Rafel, et al . 2023.
    Efcient methods for natural language processing: A survey. Transactions of
    the Association for Computational Linguistics 11 (2023), 826–860.
