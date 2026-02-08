# Research into ml techniques for translation. particularly on languages with low coverage
## Search Terms
"resource-scarce languages machine translation" , "LRL NLP" , "machine learning dead language" , "low resource natural language processing", "LRL NMT"
NMT = Neural Machine Translation
LRL = Low Resource Language
## Papers

[Machine Learning for Ancient Languages: A Survey](https://direct.mit.edu/coli/article/49/3/703/116160)

[Evaluation of open and closed-source LLMs for low-resource language with zero-shot, few-shot, and chain-of-thought prompting](https://www.sciencedirect.com/science/article/pii/S2949719124000724)
    - Focuses mostly on the eval side.
[An automated approach to identify sarcasm in low-resource language](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0307186#sec007)

[Neural Machine Translation for Low-Resource Languages from a Chinese-centric Perspective: A Survey](https://dl.acm.org/doi/full/10.1145/3665244)

[Neural Machine Translation for Low-resource Languages: A Survey](https://dl.acm.org/doi/full/10.1145/3567592#sec-5)
    - Some interesting Non-LLM approaches, but almost no LLM based ones.

[Reference-Less Evaluation of Machine Translation: Navigating Through the Resource-Scarce Scenarios ](https://www.mdpi.com/2078-2489/16/10/916)
    - ? What are they doing? they mention LoRA
    - Not useful?

[Natural language processing applications for low-resource languages](https://www.cambridge.org/core/journals/natural-language-processing/article/natural-language-processing-applications-for-lowresource-languages/7D3DA31DB6C01B13C6B1F698D4495951)
    - Links to practical papers but no direct summary of techniques.
[Breaking Through Language Barriers: A Review of OCR Technology for Low-Resource Minority Languages Based on Deep Learning](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5467297)
    - Pretty informal feeling. Double spaced.
    - Kinda dont wanna read it. Maybe if I need OCR it could be helpful.
[Optimizing Resource-Scarce Language Question Generation on Bengali : A Comparative Study of Transformer-Based Models](https://ieeexplore.ieee.org/abstract/document/11021824)
    - Uses MT5!!!
[Effectively compiling parallel corpora for machine translation in resource-scarce conditions](https://opinvisindi.is/items/b3439d0e-4a7b-4058-b86f-9e937efe3476)
    - Information on how to properly use data. Shaping it, etc.
    - Looks really really comprehensive. Also its a phd thesis.
[Navigating Complexity: A Resource-Adaptive Framework for Cross-Lingual Sentiment Analysis in Resource-Scarce Languages](https://www.researchsquare.com/article/rs-3239662/v1)
    - Gives some interesting search terms that are unique, but thats about it.

[Does learning from language family help? A case study on a low-resource question-answering task](https://www.cambridge.org/core/journals/natural-language-processing/article/does-learning-from-language-family-help-a-case-study-on-a-lowresource-questionanswering-task/AFF6EBE6285F1F1FB739CD1BD65F2A4C)
    - Specific case study on whether transfer learning can work.
    - Uses MLM, etc.
[Multilingual AI for Inclusive Language Representation: Exploring the Applications and Challenges of Transfer Learning in Low-Resource Language NLP](https://www.researchgate.net/profile/Abiodun-Okunola-6/publication/386381474_Multilingual_AI_for_Inclusive_Language_Representation_Exploring_the_Applications_and_Challenges_of_Transfer_Learning_in_Low-Resource_Language_NLP/links/674f89bc790d154bf9c27ac8/Multilingual-AI-for-Inclusive-Language-Representation-Exploring-the-Applications-and-Challenges-of-Transfer-Learning-in-Low-Resource-Language-NLP.pdf)
    - Just keywords.
[Dictionary-based Phrase-level Prompting of Large Language Models for Machine Translation](https://arxiv.org/abs/2302.07856)
    - "What if you rag'd it with a dictionary?"

[Letz Translate: Low-Resource Machine Translation for Luxembourgish](https://ieeexplore.ieee.org/abstract/document/10236754)
    - Looks pretty good and practical. Though not a survey.

[Unlocking Parameter-Efficient Fine-Tuning for Low-Resource Language Translation](https://aclanthology.org/2024.findings-naacl.263.pdf)
    - Has some different methods.

[Optimizing the Training Schedule of Multilingual NMT using Reinforcement Learning](https://aclanthology.org/2025.mtsummit-1.6/)
    - Interesting
[Reinforcement of low-resource language translation with neural machine translation and backtranslation synergies](https://www.researchgate.net/profile/Padma-Prasada/publication/382819872_Reinforcement_of_low-resource_language_translation_with_neural_machine_translation_and_backtranslation_synergies/links/66ad07d0299c327096a7632d/Reinforcement-of-low-resource-language-translation-with-neural-machine-translation-and-backtranslation-synergies.pdf)
    - Interesting
[Towards Improving Neural Machine Translation Systems for Lower-Resourced Languages: Optimising Preprocessing and Data Augmentation Techniques for English to Irish Translation ](https://www.researchgate.net/profile/Joshua-Brook-2/publication/380880902_Towards_Improving_Neural_Machine_Translation_Systems_for_Lower-Resourced_Languages_Optimising_Preprocessing_and_Data_Augmentation_Techniques_for_English_to_Irish_Translation/links/6652facdbc86444c72019c57/Towards-Improving-Neural-Machine-Translation-Systems-for-Lower-Resourced-Languages-Optimising-Preprocessing-and-Data-Augmentation-Techniques-for-English-to-Irish-Translation.pdf)
    - Word2Vec is an idea?
[Mitigating the Disparity of Machine Translation Quality for Low Resource Languages](https://umm-csci.github.io/senior-seminar/seminars/spring2023/miller.pdf)
    - Backtranslation?



## From "Natural Language Processing with Transformers" book
    - https://amitness.com/posts/data-augmentation-for-nlp

## Search Terms
"few shot learning translation", "few shot learning translation survey", "few shot learning "neural machine translation" survey"
## Papers
[The Unreasonable Effectiveness of Few-shot Learning for Machine Translation](https://proceedings.mlr.press/v202/garcia23a.html)

[Few-Shot Learning Translation from New Languages](https://aclanthology.org/2025.emnlp-main.163/)

[Survey of Low-Resource Machine Translation](https://aclanthology.org/2022.cl-3.6.pdf)
    - Autoencoder backtranslation for monolingual data? (3.2.1)
    - Unsupervised MT, doing stuff with word embeddings across the languages (3.2.2)
    - 3.3 pre-trained embeddings
    - 5.2 bilingual lexicons

[Low-resource Neural Machine Translation: Methods and Trends](https://dl.acm.org/doi/full/10.1145/3524300#sec-3)