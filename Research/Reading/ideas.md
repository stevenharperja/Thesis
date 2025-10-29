1. Training a model to have low rank matrices.
    -   Related Work:
        -   Wei Wen, Cong Xu, Chunpeng Wu, Yandan Wang, Yiran Chen, and Hai Li. Coordinating filters for
            faster deep neural networks. In Proceedings of the IEEE international conference on computer
            vision, pp. 658–666, 2017
            https://wenwei202.github.io/
            https://openaccess.thecvf.com/content_ICCV_2017/papers/Wen_Coordinating_Filters_for_ICCV_2017_paper.pdf
        -   R2 Loss: Range Restriction Loss for Model Compression and Quantization
            https://machinelearning.apple.com/research/range-regularization
            https://arxiv.org/pdf/2303.08253
        -   Huanrui Yang, Minxue Tang, Wei Wen, Feng Yan, Daniel Hu, Ang Li, Hai Li, and Yiran Chen. Learning low-
            rank deep neural networks via singular vector orthogonality regularization and singular value sparsification,
            2020. https://arxiv.org/abs/2004.09031
            -   LASER paper noted that ". [Yang et al.,
                2020] have enforced low-rank-ness of weight matrices for the purposes of memory efficiency, but the resulting
                models fail to achieve performance equivalent to their overparametrized counterpart"
                -   So perhaps it is not worth pursuing.
                -   Perhaps I could instead do it as a retraining problem?
                    -   Use Knowledge distillation to train it and train it to have low rank?
                        But knowledge distillation takes too long to train :/
                        -   Maybe instead I could do something funky, and use the LoRA paper's thing to just train a lower rank set of stuff, 
                            to try to eliminate the ranks of the original matrix? unfortunately I could only eliminate as many ranks as I have i think. So thats no good for eliminating lots of ranks like we want. :/
                            -   There apparently exist some more gradual versions of LoRA which allow a rank to be learned? maybe I could add the rank parameter to this to try to minimize the rank of the result????? but that still migth not fix it?
                                -   How do these work?
                            -Could do just low rank factorization, then add lora on top.
            -   They use the nuclear norm here, calculating it fully using SVD. But the norm I was thinking of doesn't do that. 
                Maybe it could be a good alternative? I should do some math proofs on it or something.
                -   https://ee227c.github.io/code/lecture5.html Maybe nuclear norm is the existing standard???
                -   https://math.stackexchange.com/a/3415724 Remember that singular values would get funky due to multiplications.
                -   https://www.aimsciences.org/article/doi/10.3934/jimo.2022045 They introduce a different rank regularization technique
                    -   They say that the nuclear norm has problems where its dominated by the large singular values too much?
                -   To prove the other one, or that the frobenius norm would be a good alternative, I think I need to prove this property:
                    -   "The nuclear norm ‖ A ‖∗ is a convex envelope of the rank function rank ( A ) , so it is often used in mathematical optimization to search for low-rank matrices. " - https://en.wikipedia.org/wiki/Matrix_norm
        -   Yuhui Xu, Yuxi Li, Shuai Zhang, Wei Wen, Botao Wang,
            Yingyong Qi, Yiran Chen, Weiyao Lin, and Hongkai Xiong.
            Trained rank pruning for efficient deep neural networks.
            arXiv preprint arXiv:1812.02402, 2018. 1, 2, 3, 4, 5, 6, 11
        -   Jose M Alvarez and Mathieu Salzmann. Compression-aware
            training of deep networks. In Advances in Neural Informa-
            tion Processing Systems, pages 856–867, 2017. 2
                

2. Treating all of a model's weights as a tensor then decomposing the tensor to make it so there are less parameters to train
    -    Parameter sharing mixed with Low Rank Approximation.

3. Combining techniques just to apply them
    - maybe quantization and LRF and LoRA on the gradients and maybe ESPACE on the activation functions. See how small I can make one.
    - Is it possible to get GPT-2 or 3 to run on 4 GB or so? then it would fit on my laptop.

4. Combining ideas 1 and 2 somehow? training a low rank tensor?

5. Making a JP version of grammarly?

6. Making an RL model for board games or video games?

7. Look up job descriptions which list model compression or its techniques as a prerequisite, and try doing the job they list if they list specific projects?
    -    See C:\Users\shwes\Documents\school\Thesis\Research\Reading\Real industry ML.md

8. Implementing a RAG model:
    - Steps:
        1. Understand RAG
        2. Understand Agentic RAG
        3. Look up company API's or something
        4. Find a use case to apply it to as thesis. (doesn't have to be about model compression perhaps)

10. Don't do mathematical research thing because the results are not guaranteed.

11. Asking sacred heart if they have anything they need analyzed for medical research  https://www.providence.org/locations/wa/sacred-heart-medical-center/research

X12. I wonder if I could combine RAG to use image retrieval instead of text retrieval, and pipe it into chatgpt or something. then have it describe or create a similar image?
    -   Or maybe a diffusion image production network could be used to query for images from an image database?
        -   text -[premade diffusion model]-> image -[premade image embedding model]-> image search using a vector database
        -   Maybe it already exists? Maybe its a good alternative to tagging images? Either way its fairly simple
            -   https://www.yeschat.ai/gpts-9t55QeOYvWW-Descriptive-image-search This is NOT it?
        -   https://www.pinecone.io/learn/clip-image-search/ Could be a good model. Actually this thing just does the whole process I think?
        -   https://milvus.io/docs/text_image_search.md Image to text embeddings might be way better lol.

13. RAG Use cases:
    -   When would I want a vector database of pieces of text?

# 14. Federated Learning project
I want the focus to be on the federated learning and compressed model training aspects. Distributed training of a model on phones.

-   https://medium.com/@entrepreneurbilal10/federated-learning-95d7a6435f08 
-   https://www.codegenes.net/blog/pytorch-federated/ 
-   https://medium.com/@mayurkoshti12/deploying-pytorch-models-to-mobile-a-step-by-step-guide-for-ios-and-android-with-pytorch-mobile-f9ba03a3c34a 
-   Based on jobs like this: https://www.indeed.com/viewjob?jk=4be6baf9a14d5d85&from=shareddesktop_copy 
-   Stuff to use:
    -   Tensorflow Lite
        -   https://www.influxdata.com/blog/tensorflow-lite-tutorial-how-to-get-up-and-running/
    -   ML Kit (including GenAI APIs)
    -   MediaPipe
    -   PyTorch Mobile
-   Vague project outline:
    -   Design, develop, and deploy on-device machine learning models optimized for Android, ensuring low latency and minimal resource consumption.
    -   Implement local signal aggregation and real-time pattern recognition logic to enable responsive in-app actions driven by on-device inference.
    -   Architect systems that support telemetry, secure logging, and privacy-first feedback collection for monitoring and evaluation.
    -   Apply model compression and optimization techniques (e.g., quantization, pruning, distillation) to meet mobile performance constraints.
    -   Develop secure, privacy-first solutions where all data processing and ML inference occur strictly on-device, with no external data exposure.
    -   Enable mechanisms for continuous local learning and model updates using device-resident data and signals, without compromising privacy.
    -   Ensure integration with Android’s security model and collaborate with platform and product teams to deploy AI features safely at scale.
-   Should also cause me to learn these skill points:
    -   Proven experience in Android development (Kotlin/Java), with strong understanding of system architecture, resource management, and performance tuning.
    -   Hands-on expertise with on-device ML frameworks including TensorFlow Lite, ML Kit, MediaPipe, and PyTorch Mobile.
    -   Solid foundation in machine learning and signal processing techniques, such as time-series modeling, clustering, classification, and real-time event detection.
    -   Strong knowledge of mobile data handling and Android security practices, including permissions, sandboxing, and secure data storage.
    -   Understanding of privacy-preserving learning techniques and data governance in mobile environments.
    -   Familiarity with secure data handling on Android, including encrypted storage, permissions, sandboxing, and secure compute enclaves.
    -   Experience with telemetry systems and evaluation pipelines for monitoring model performance on-device at scale.
-   Additionally cause me to learn these skill points:
    -   Experience building ML-driven mobile applications in domains requiring user personalization, privacy, or security.
    -   Understanding of real-time data processing and behavioral modeling on resource-constrained edge devices.
    -   Knowledge of on-device learning techniques, federated learning, or personalization methods.
    -   Prior contributions to systems using federated learning, differential privacy, or local fine-tuning of models is a plus
    -   Experience with backend infrastructure for model management (e.g., model registries, update orchestration, logging frameworks) is a plus.
    -   Prior work with anomaly detection or behavioral modeling in resource-constrained environments is a plus.
    -   Experience developing responsive systems capable of monitoring local context and dynamically triggering actions based on model outputs is a plus
    -   Experience optimizing models for ARM architectures is a plus
### Project ideas:
-   Image describer on mobile, distributed on 3 virtual Android phones.
-   Image classifier on mobile (maybe a plant recognition app or something?)
-   local GPT  
    -   use that 1.5 bit thing?
    -   Squish something down? isnt there a thing called Mobilenet? MobileGPT? surely there is something even if its bad.
-   I want the focus to be on the federated learning and compressed model training aspects.
-   https://medium.com/@entrepreneurbilal10/federated-learning-95d7a6435f08 Something based on this?

-   Federated learning analysis of toy medical data from a public dataset online.

todo next: 
-   Find tutorials on federated learning, Pytorch Mobile, Tensorflow Lite and federated learning, etc.
-   See what ML Kit and MediaPipe are.
-   Try the above tutorial on Flower.
-   After checking those out, if it still sounds viable, ask Bojian his thoughts on if it would be viable.
    -   Also ask him his thoughts on Phd for employability?
-   Try this https://flower.ai/docs/framework/tutorial-series-get-started-with-flower-pytorch.html 

-   Sources to look at:
    -   https://flower.ai/docs/framework/tutorial-series-what-is-federated-learning.html
        -   Model soup 2 babyeeeee Communication-Efficient Learning of Deep Networks from Decentralized Data https://arxiv.org/abs/1602.05629
        -   https://en.wikipedia.org/wiki/Differential_privacy 
    -   I wanna look back at that LoRA survey paper again.
    -   Data to use: https://research.ewu.edu/computer_science/data 
        -   Perhaps focus on medical data for diagnoses?
            -   Image data?
            -   EEG data? https://datasetsearch.research.google.com/search?src=0&query=phone%20medical&docid=L2cvMTFsdno5MjhieA%3D%3D
        -   mobile phone focused data:
            -   Malware?            
                -   https://figshare.com/articles/dataset/Android_malware_dataset_for_machine_learning_2/5854653?file=10391991 
                -   https://zenodo.org/records/3632184
            -   Object detection?   https://universe.roboflow.com/yolov5appium/android-views 
            -   RL button manipulation?
                -   https://huggingface.co/datasets/Tonic/android-operator-episodes
                -   https://figshare.com/articles/dataset/Dataset_of_smartphone-based_finger_tapping_test/26940823?file=49014607

            -   Application transmitted data?
                -   https://datasetsearch.research.google.com/search?src=0&query=android&docid=L2cvMTF2eXE0Z2h0NA%3D%3D
            -   Qualitative medical data?
                -   This is questionably valid. https://defined.ai/datasets/medical-app-analytics
        KAGGLE HAS BETTER DATA :
        - Medical Idea
            -   Sleep disorder data https://www.kaggle.com/datasets/varishabatool/disorder
            -   Chest XRays? https://www.kaggle.com/datasets/muhammadrehan00/chest-xray-dataset
            -   Brain tumors https://www.kaggle.com/datasets/ishans24/brain-tumor-dataset
            -   College placement https://www.kaggle.com/datasets/vrajesh0sharma7/college-student-placement
            -   General synthetic healthcare dataset https://www.kaggle.com/datasets/prasad22/healthcare-dataset 
            -   Heart failure dataset (Also has links to similar datasets) https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction
            -   Diabetes https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset
            -   General https://www.kaggle.com/datasets/itachi9604/disease-symptom-description-dataset 
        - Phone Idea
            -   Flower images https://www.kaggle.com/datasets/abedano/iris-flower-specie-dataset
            -   Clouds https://www.kaggle.com/datasets/jockeroika/clouds-photos
            -   Music Preferences? https://www.kaggle.com/datasets/nelgiriyewithana/top-spotify-songs-2023
            -   Twitter Sentiment Analysis? https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis
            -   Network security? https://www.kaggle.com/datasets/dhoogla/nfunswnb15v2
            -   Face Recognition https://www.kaggle.com/datasets/jessicali9530/lfw-dataset
            -   Human activity recognition https://github.com/xmouyang/FL-Datasets-for-HAR !!!!!!!!!!!!!!!!!!!!!!
                -   Health related too! https://github.com/xmouyang/FL-Datasets-for-HAR/tree/main/datasets/IMU 
            -   Smaller managable Images and text collection https://leaf.cmu.edu/
            -   large Images https://machinelearning.apple.com/research/flair
            -   Object Detection https://github.com/MenguChen/Federated_object_detection 
        - Company Idea
            -   Employee https://www.kaggle.com/datasets/tawfikelmetwally/employee-dataset
        - General FL
            -   A collection! https://www.kaggle.com/datasets/wonghoitin/datasets-for-federated-learning 
            -   Another collection! https://flower.ai/docs/datasets/ 
            -   satellite pictures FL specific https://arxiv.org/abs/2505.08325
        
            
    -   Tensorflow Federated https://www.geeksforgeeks.org/deep-learning/federated-learning-with-tensorflow-federated/
    -   https://www.ibm.com/think/topics/federated-learning

    -   https://flower.ai/docs/examples/
        -   Maybe learning with Jax could be fun https://flower.ai/docs/examples/quickstart-jax.html 
            -   https://docs.jax.dev/en/latest/jax-101.html#jax-101
    -   Android example!
        -   https://flower.ai/docs/examples/android.html !!!!!!!!!!!!!!!!!
        -   https://flower.ai/docs/examples/app-pytorch.html

-   Perhaps I go with the phone idea, and train a convnet using the phone hardwares (not sure how itll do on an emulator)
    -   Use model compression and see what exists already with flower or other architectures?
-   Do we still have those raspberry pis? I could try this: https://flower.ai/docs/examples/embedded-devices.html
    -   Ask Bojian and maybe ask Stu?
    -   I really like this idea. I think its so cool.



-   Raspberry pi idea related research:
    -   https://github.com/SonySemiconductorSolutions/aitrios-rpi-tutorials-ai-model-training
    -   https://medium.com/@wongsirikuln/yolo-model-compression-via-filter-pruning-for-efficient-inference-on-raspberry-pi-c8e53d995d81
    -   https://colab.research.google.com/github/SonySemiconductorSolutions/aitrios-rpi-tutorials-ai-model-training/blob/main/notebooks/mobilenet-rps/custom_mobilenet.ipynb