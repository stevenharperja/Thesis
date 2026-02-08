# Some techniques/tools/perspectives found from researching how others have done ML on dead languagess

### From "Natural Language Processing with Transformers" book
- Fill Mask/ Masked Language Modelling pg. 290
- Data Augmentation: pg. 272
    - Back translation. of source text eg. En -> German -> En
    - Token Preturbations. (words or tokens?)
        - Synonym replacement
        - Random Insert
        - Random swap
        - Random Deletion
    - Link to https://amitness.com/posts/data-augmentation-for-nlp
        - Synonym replacement
            - Thesaurus
            - Word Embeddings
            - Masked language modelling from a pretrained model to predict similar words
            - low TF-IDF based word replacement. [(term frequency–inverse document frequency)](https://en.wikipedia.org/wiki/Tf%E2%80%93idf)
        - Back Translation
        - Text Surface Transformation //probably not useful
        - Random noise injection
            - Spelling error injection //probably not useful
            - querty keyboard error injection //probably not useful
            - Unigram Noising, Based on frequency distribution of a word
            - Blank noising, replacing words with "_" or some other thing.
            - Sentence Shuffling
            - Random Insert
            - Random Swap
            - Random Delete
        - Instance Crossover Augmentation, (swap pieces of different data phrases within same class)
        - Syntax tree manipulation, eg. active voice to passive voice
        - MixUp for Text 
            - //Used for classification but could probably be adapted to translation? maybe??? HMM maybe not actually. Though Maybe I could pretrain on a classification task.
            - word embeddings 
            - sentences
        - Generative methods
            - Conditional Pre-trained language models
                - Take a starter "hook" such as a class label, and have a gpt-2 or other model print out a coressponding sentence that fits that label.
        - Links are provided to libraries. and the book does this too.
    - Unsupervised Data Augmentation (pipeline technique?) pg. 295
    - Uncertainty-aware self-training pg. 296
    - Keywords to look up next: "Few shot learning" "few shot translation" etc.

            

