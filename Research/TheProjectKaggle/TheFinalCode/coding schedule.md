- Create inference code for showing results
- Choose 10-20 things to test:
Answer 15 questions. 
- Get these from the future work sections to start with.
    - From 1st place: 
        - "Due to the extreme time constraints towards the end of the competition, we unfortunately ran out of time to conduct an ablation study on the specific performance gain of this weighted model within the ensemble, but this generally outlines our underlying rationale and approach." 
            - NEVERMIND they did it already.
        - none
    - 2nd:
        - none // key takeaway was to get more data.
    - 3rd:
        - none //they used an LLM (qwen) to pick their ensemble output????? they literally grab the probabilities from the output distrib of the LLM though which is an interesting way to do it.
    - 4th:
        - //didn't use train csv at all. but it was included anyways
        - "The model struggled to generate short translations for very short transliterations. It also sometimes struggled with capitalization of proper nouns (PNs)."
        - "In the final moments of the competition (a day to the deadline) it became obvious to us that we were bottlenecked by data (as we observed that in some few examples in published_texts.csv, our model struggled with capitalization of proper nouns). We then decided to source more data, but unfortunately we were handicapped by compute. Although we couldn’t train on our new training data, we will detail how we curated it:
        
        We chose to include all the 7953 transliterations contained in the published_texts.csv. First of all, we filtered out all the training set documents from this leaving us with 6392 documents. We then split each document that was longer than 500 characters into chunks. This was done by first splitting the text by space, and concatenating the words until we reached the character limit (500). This process yielded 9237 samples. We then used our strongest model to translate all the samples. These translations were then fed into Gemini 3.1 Flash via AI studio. Gemini was prompted to evaluate the quality of each translation using the transliteration as context. It was instructed to only fix a translation if it evaluated it to be of poor quality. It was also instructed to do the following: follow the model’s language style (which mirrored human translators) and avoid robotic and overly complex grammar; correct capitalization in personal names; etc."
    - 5th:
        - none
    - 6th:
        - none
    - 7th:
        - managed to fit byt5-xl fp32 on P100 for inference, so maybe look into that
        - none - no future work listed
    - 8th:
        - //mainly stuck to data the hosts gave
        - none
    - 9th:
        - //extracted with gemini, has good detail about it. probably one of the best writeups not in 1st-4th. I should write like them if possible.
        - "With more time, cleaning the noisiest extracted pairs and retraining may have unlocked the larger model's potential — the 3rd place team used ByT5-Large and XL successfully with cleaner synthetic data." //can't do
        - "With more GPU time, approaches like cross-attention warmup schedules or the 3rd place team's supervised CPT strategy (using seq2seq synthetic drills rather than unsupervised text) may have resolved the misalignment — but there wasn't enough compute budget to iterate further."
    - 10th:
        - none
    - 11th:
        - none
    - 12th:
        - //small working capitol. might be a good one to imitate.
        - didn't have much time to get as much data extraction done as they wanted.
    - 13th:
        - //good explanation of why they had to get better data. why the given data wasn't good enough.
        - none
    - 14th:
        - commenter: "I have one question: How did you fit all online-trained models in Kaggle's 2×T4 inference budget?"    
            - reply: "Inspired by the Jigsaw competition, you can use the Unsloth framework to train 14B-sized models on a T4 GPU."
        - none
    - 15th:
        - none

- make "does this improve stuff in my results?" for every technique I try (will yield about 8)



Techniques from the spreadsheet "Whatworkeddidntwork.xlsx" that I want to integrate into those tests
- MBR from one model
- MBR using an ensemble
- K-fold cross validation for ensemble
- length tuning
- R-Drop
- Projected Gradient Descent
- fine-tuning on train.csv after using other datasets
- imitate 3rd place's grammar generation results using local Ollama model?
- //using an LLM to automate my work extensively.
- 
- 
- 
- 
- ensembling
- k-fold cross-validation

Problems to address:
- Knowing what they've done now, and using the published data. What's the best you could do to replicate their results on basically no budget?
    - Is this even worth asking?
- Combine 12th place's solution with good data from other folks?
- Train on 4th places unused data?
- Try using 3rd places techniques on 9th places techniques?

Stuff I would want to try out that I think is cool:
- MBR
- 3rd place's LLM excersizes approach
- 3rd place's qwen as ensemble evaluator approach
- 9th place's extraction techniques
- an ablation study if one of them doesn't have one. ie. choosing one model of an ensemble to evaluate just that one and building a chart
    - 4th
    - 8th