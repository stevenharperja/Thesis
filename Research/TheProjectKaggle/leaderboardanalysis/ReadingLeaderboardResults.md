- [1st place](https://www.kaggle.com/competitions/deep-past-initiative-machine-translation/writeups/dpc-1st-data-quality-dictates-everything)
    - https://www.kaggle.com/code/ngyzly/better-candidate-diversity-on-public-model?scriptVersionId=302266718
        - [MBR](https://suzyahyah.github.io/bayesian%20inference/machine%20translation/2022/02/15/mbr-decoding.html_)
            - Random binge of motivation/mindset blogposts by this person. https://suzyahyah.github.io/categories
        - [MBR](https://www.youtube.com/watch?v=w8_rLLPkE5A)
            - "the center-est (vector) point of a lot of (weighted by stuff like probability (like beam search)) output samples. Choose the output which is closest to that." 
        - [Temperature](https://stackoverflow.com/questions/58764619/why-should-we-use-temperature-in-softmax/63471046#63471046)
        - "Mbr seems to literally be just pick the output of the models in the ensemble which have good bleu, jaccard, chrf++, etc. scores when their text is (n^2) compared with each other's texts."
    - notable things:
        - They recreated their own dataset.
        - Their artificial data was completely made by an *LLM* piecing things together, NOT a translation model.
        - They took MBR pointers from the MBR notebook publicly shared near the end of the kaggle competition.
    
- [2nd place](https://www.kaggle.com/competitions/deep-past-initiative-machine-translation/writeups/2nd-place-data-centric-akkadian-nmt)
    - Not that interesting?
- [3rd place](https://www.kaggle.com/competitions/deep-past-initiative-machine-translation/writeups/3rd-synthetic-data-to-teach-oa-fundamentals)
    - "THEY MADE CLAUDE CODE TEACH AKKADIAN!!!"

- [6th place](https://www.kaggle.com/competitions/deep-past-initiative-machine-translation/writeups/dpc-6th-solution)
    - "JACK MADE IT AND ALL THE TEAM WERE SILVER BEFORE!! OH THE DRAMA!!"
    - They do model training tricks yippee!
    - Their cleaned Train.csv yielded 34.0??????
        - (I'm assuming model tricks applied).
    - they said Model weight averaging performed poorly
    - bidirectional training apparently went unexplored, but didn't look promising.
    - [Open source ocr method](https://www.kaggle.com/code/angantyr/text-extraction-from-pdf-docs-using-glm-oc)
- [7th place](https://www.kaggle.com/competitions/deep-past-initiative-machine-translation/writeups/short-7th-place-note)
    - bigger models were better
    - byt5-xl 3 epochs

- [5th place](https://www.kaggle.com/competitions/deep-past-initiative-machine-translation/writeups/5th-solution)
    - byt5-xl
    - Synthetic data generated via pseudo-labeling and back-translation
        1. stg. 1
            -   Continued pre-training — T5 span corruption on transliterations from published_texts.csv.
            -   Fine-tuning on EvaCun.
            -   Fine-tuning on PDF data.
            -   -----------
            -   A reverse model (English → Akkadian) is also trained with the same pipeline by swapping transliteration and translation.
        2. stg. 2
            -   Using the reverse model from Stage 1, we generate additional training pairs:
            -   -----------
            -   Pseudo-label: translate published_texts.csv transliterations to English with the forward model using beam search.
            -   Back-translation: first generate 10,000 translation-like English sentences with Qwen3.5-27B, using translations from both train.csv and the PDF-extracted data as few-shot examples, and then back-translate them into Akkadian with the reverse model.
    - quantization for inference.

- [8th place](https://www.kaggle.com/competitions/deep-past-initiative-machine-translation/writeups/8th-place-solution-2-stage-fine-tuning-high-qua)
    - byt5-xl
    - SFT?
        - https://www.geeksforgeeks.org/artificial-intelligence/supervised-fine-tuning-sft-for-llms/
    - did NOT make their own dataset!!!!
    - Uniquely? used RL, though it didn't help
    - GLM ocr
- [10th place](https://www.kaggle.com/competitions/deep-past-initiative-machine-translation/writeups/10th-place-solution-seq2seq-cpt-pseudo-label)
    - byt5-large, byt5-xl, MADLAD-400-3B-MT
        - What??? Unique!!!
        - Madlad does pretty well apparently.
- [11th place](https://www.kaggle.com/competitions/deep-past-initiative-machine-translation/writeups/11th-place-solution)
    - pretrained on *language-pairs*; labelled data! different.
    - no external data sources, their LLM processes publications.csv and is easy enough to follow
        - I want to look into how they did it so I can learn to do the same text extraction automation. I think it would be very very useful
    - EMA? (review for me but I forgot the acronym reading)
        - https://medium.com/@heyamit10/exponential-moving-average-ema-in-pytorch-eb8b6f1718eb
    - I'm surprised they also (as did some other leaderboard peeps) used the dictionary data for pretraining. I felt so silly when it didn't work.
    - Minimum risk training? (didnt work)
        - https://arxiv.org/abs/1512.02433
- [12th place](https://www.kaggle.com/competitions/deep-past-initiative-machine-translation/writeups/12th-place-notebooklm-and-single-byt5-base-model)
    - A SINGLE byt5-base model. NO ENSEMBLE?????? their techniques must be goated
    - NotebookLM for easy text extraction.
    - didn't do much with byt5-large
    - Used external data
    - Synthetic data is interesting and the build process was custom coded for the dataset.
    - Literally just "get the data and feed it into the model in a specific order"




takeaways:
- It's good to explore with a smaller model version, then try out a bigger version to see if it works better. not sure when this would be the case that it would work better though.
- both 8 and 12 (and maybe others) didn't use any resources which costed money. so its perfectly reasonable to make gold without shelling out money for api calls.