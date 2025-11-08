
- Jax https://docs.jaxstack.ai/en/latest/getting_started.html


### Misc papers
-    Towards Federated Learning on Fresh Datasets "Updating datasets over time" https://ieeexplore.ieee.org/document/10298589
-   Federated Learning from Small Datasets   https://arxiv.org/abs/2110.03469



- Flower Intro https://flower.ai/docs/framework/tutorial-series-what-is-federated-learning.html
    -   What is secure aggregation?
        -   Practical Secure Aggregation for Privacy-Preserving Machine Learning https://dl.acm.org/doi/10.1145/3133956.3133982
    -   Differential Privacy? https://flower.ai/docs/framework/explanation-differential-privacy.html
        -   Model Inversion Attacks https://www.tillion.ai/blog/model-inversion-attacks-a-growing-threat-to-ai-security
            -   I wonder if a model could be trained on fully encrypted data and still produce decent outputs? probably not. but maybe.
                -   https://www.ericsson.com/en/blog/2021/9/machine-learning-on-encrypted-data
                -   Machine Learning Training on Encrypted Data with TFHE https://dl.acm.org/doi/10.1145/3643651.3659891
                -   Machine Learning Meets Encrypted Search: The Impact and Efficiency of OMKSA in Data Security https://onlinelibrary.wiley.com/doi/full/10.1155/int/2429577
                    -   https://en.wikipedia.org/wiki/Searchable_symmetric_encryption
                -   https://csrc.nist.gov/CSRC/media/Projects/pec/stppa/stppa-03-kristin-Private-AI.pdf
                    -   https://en.wikipedia.org/wiki/Homomorphic_encryption
                        -   https://en.wikipedia.org/wiki/Predictive_analytics 
                        -   https://en.wikipedia.org/wiki/Ring_learning_with_errors
                        
                        -   Recent advances of privacy-preserving machine learning based on (Fully) Homomorphic Encryption https://sands.edpsciences.org/articles/sands/abs/2025/01/sands20240021/sands20240021.html

### Surveys:
#### IOT
-   Federated Learning for IoT: A Survey of Techniques, Challenges, and Applications  https://www.mdpi.com/2224-2708/14/1/9
    -   https://en.wikipedia.org/wiki/Secure_multi-party_computation
        -   https://en.wikipedia.org/wiki/Yao%27s_Millionaires%27_problem
            -   Practical card-based implementations of Yao's millionaire protocol https://www.sciencedirect.com/science/article/pii/S0304397519307042?via%3Dihub
            -   Zero Knowledge Proof https://en.wikipedia.org/wiki/Zero-knowledge_proof 
    -   Li, S.; Ngai, E.C.H.; Voigt, T. An experimental study of byzantine-robust aggregation schemes in federated learning. IEEE Trans. Big Data 2023, 10, 978–988. https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10018261 
    -   Liu, S.; Chen, Q.; You, L. Fed2a: Federated learning mechanism in asynchronous and adaptive modes. Electronics 2022, 11, 1393. 
        -   What is FedAsync? 
            -   https://arxiv.org/abs/1903.03934
    -   Privacy-Preserving Asynchronous Federated Learning Framework in Distributed IoT Yan, X.; Miao, Y.; Li, X.; Choo, K.K.R.; Meng, X.; Deng, R.H. Privacy-preserving asynchronous federated learning framework in distributed iot. IEEE Internet Things J. 2023, 10, 13281–13291.
        -   Blockchain
    -   SMPC, HE
        -   79 Fang, H.; Qian, Q. Privacy preserving machine learning with homomorphic encryption and federated learning. Future Internet 2021, 13, 94.  https://www.mdpi.com/1999-5903/13/4/94
#### HE PPML
-   Recent advances of privacy-preserving machine learning based on (Fully) Homomorphic Encryption https://sands.edpsciences.org/articles/sands/abs/2025/01/sands20240021/sands20240021.html
    -   

-   A comprehensive survey on secure healthcare data processing with homomorphic encryption: attacks and defenses https://link.springer.com/article/10.1186/s12982-025-00505-w
    -   https://en.wikipedia.org/wiki/Chosen-plaintext_attack
    -   https://en.wikipedia.org/wiki/Fault_injection
    -   Lattice attack?
    -   Partially vs Fully homeomorphic encryption is only the difference of what operations can be done, not how encrypted it is.

-   Approximate homomorphic encryption based privacy-preserving machine learning: a survey https://link.springer.com/article/10.1007/s10462-024-11076-8
    -   single instruction multiple data?
        -   https://en.wikipedia.org/wiki/Single_instruction%2C_multiple_data
    -   Paillier P (1999) Public-key cryptosystems based on composite degree residuosity classes. In: International
Conference on the Theory and Applications of Cryptographic Techniques, pp. 223–238. Springer
        -   https://en.wikipedia.org/wiki/Paillier_cryptosystem
        -   CKKS?
            -   https://www.youtube.com/watch?v=iQlgeL64vfo&themeRefresh=1
        -   DP
            -   Xu Z, Collins M, Wang Y, Panait L, Oh S, Augenstein S, Liu T, Schroff F, McMahan HB (2023) Learning
to generate image embeddings with user-level differential privacy. In: Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition, pp. 7969–7980
            -   Guan J, Fang W, Huang M, Ying M (2023) Detecting violations of differential privacy for quantum algo-
rithms. In: Proceedings of the 2023 ACM SIGSAC Conference on Computer and Communications
Security, pp. 2277–2291
            -   Shi Y, Yang Y, Wu Y (2024) Federated edge learning with differential privacy: An active reconfigurable intel-
ligent surface approach. IEEE Transactions on Wireless Communication
        -   SMPC

        -   HE ML examples
            -   Choi H, Woo SS, Kim H (2024) Blind-touch: Homomorphic encryption-based distributed neural network
inference for privacy-preserving fingerprint authentication. In: Proceedings of the AAAI Conference on
Artificial Intelligence, vol. 38, pp. 21976–21985
            -   Kim D, Guyot C (2023) Optimized privacy-preserving cnn inference with fully homomorphic encryption.
IEEE Trans Inf Forens Secur 18:2175–2187
            Question: Wait, doesn't using HE for ML not solve the problem of model-inversion attacks at all? What is it useful for?
            -   Hijazi NM, Aloqaily M, Guizani M, Ouni B, Karray F (2023) Secure federated learning with fully homomor-
phic encryption for iot communications. IEEE Internet of Things Journal https://ieeexplore.ieee.org/abstract/document/10208145 <---Now that sounds cool
                -   I think the idea is to use HE on the model gradients and models themselves?
                -   26. H. Fang and Q. Qian, “Privacy preserving machine learning with homomorphic encryption and federated learning,” Future Internet, vol. 13, no. 4, p. 94, 2021.
                    -   Shokri, R.; Stronati, M.; Song, C.; Shmatikov, V. Membership inference attacks against machine learning models. In Proceedings
of the 2017 IEEE Symposium on Security and Privacy (SP), San Jose, CA, USA, 22–26 May 2017; pp. 3–18 https://ieeexplore.ieee.org/document/7958568 
                -   28. N. Wang, “A blockchain based privacy-preserving federated learning scheme for Internet of Vehicles,” Digit. Commun. Netw., vol. 9, May 2022.

#### Secure Multi-Party Computation for ML

#### Model-inversion attacks how tos
-   Hacking deep learning: model inversion attack by example https://blogs.rstudio.com/ai/posts/2020-05-15-model-inversion-attacks/
    -   Question: So from what I understand, using HE on the model, or on the data, wouldn't help at all? im confused by HE     
    -   Pretty cool. Differential privacy does have some effect, though I would be interested in seeing an experimental comparison of doing a model inversion attack on two models with about the same accuracy on train/val sets, but one which has had DP done to it.
        -   I could modify the code they have myself to see, it wouldn't be too hard probably.
    -   "Now, as the adversary won’t call the complete model, we need to “cut off” the second-stage layers. This leaves us with a model that executes stage-one logic only. We save its weights, so we can later call it from the adversary:"
        -   I think this means that they get a latent intermediate vector output from within the model. So this would require knowing the model as a whole. So maybe HE could fix this? idk how though.