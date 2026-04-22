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
- [4th place](https://www.kaggle.com/competitions/deep-past-initiative-machine-translation/writeups/fourth-place-solution-writeup)
    - models:
        - mT5-Large / ByT5-Large
        - ByT5-XL (bfloat16)
    - No synthetic data? just good data cleaning and splitting and gathering.
    - MBR ensemble
        - ONLY chrf++
    - weighted averaging of weights of different model checkpoints.
    - Has section for further work, which was a data synthesis method.
    - Lotsa posted code

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
- [6th place](https://www.kaggle.com/competitions/deep-past-initiative-machine-translation/writeups/dpc-6th-solution)
    - "JACK MADE IT AND ALL THE TEAM WERE SILVER BEFORE!! OH THE DRAMA!!"
    - They do model training tricks yippee!
    - Their cleaned Train.csv yielded 34.0??????
        - (I'm assuming model tricks applied).
    - they said Model weight averaging performed poorly
    - bidirectional training apparently went unexplored, but didn't look promising.
    - [Open source ocr method](https://www.kaggle.com/code/angantyr/text-extraction-from-pdf-docs-using-glm-oc)
    - lotsa posted code
- [7th place](https://www.kaggle.com/competitions/deep-past-initiative-machine-translation/writeups/short-7th-place-note)
    - bigger models were better
    - byt5-xl 3 epochs


- [8th place](https://www.kaggle.com/competitions/deep-past-initiative-machine-translation/writeups/8th-place-solution-2-stage-fine-tuning-high-qua)
    - byt5-xl
    - SFT?
        - https://www.geeksforgeeks.org/artificial-intelligence/supervised-fine-tuning-sft-for-llms/
    - did NOT make their own dataset!!!!
    - Uniquely? used RL, though it didn't help
    - GLM ocr
    - vLLM  to host llm
- [9th place](https://www.kaggle.com/competitions/deep-past-initiative-machine-translation/writeups/9th-place-solution-for-the-deep-past-challenge-c)
    - FULL CODE ON GITHUB!!!!!
    - A very very in-depth write-up
    - 4 ensemble. one constituent is a soup of checkpoints (average of model weights)
    - Beam search but use MBR
    - Does post-processing
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
- [13th](https://www.kaggle.com/competitions/deep-past-initiative-machine-translation/writeups/top-13-solution)
    - inference output post processing
    - MBR reranking of beam outputs. but only one model, not ensemble; Single-model MBR
    - good detail on why byt5 is a good choice of model. 
    - Long explanation
    - Trained somemthing to normalize text??
    - sample inferance with beams and multiple temperatures
    - MBR would be a good and easy technique to take from this.
        - They provide charts of values for ablation
    - They did an ensemble and it was slightly worse.
    - "in low-resource translation with noisy historical text, data construction and normalization can matter more than model scaling"
- [14th](https://www.kaggle.com/competitions/deep-past-initiative-machine-translation/writeups/llm-online-learning-is-all-you-need)
    - tried using models: qwne3,qwen2.5, phi-4, byt5
    - tried RL (reward hacking?)?
    - training models on each other's outputs helped. and iterated that. "online learning"
    - Used llms?
- [15th](https://www.kaggle.com/competitions/deep-past-initiative-machine-translation/writeups/15th-place-gold-solution)
    - final submission consisted of an MBR ensemble of ByT5 Small, ByT5-Base, Qwen3-14B, and Qwen3-4B-Instruct models using sacrebleu.metrics.CHRF(word_order=2).
- [19th](https://www.kaggle.com/competitions/deep-past-initiative-machine-translation/writeups/20th-place-byt5-base-span-corruption-synth-and-s)
    - model soup & ensemble. more data finding
- [20th](https://www.kaggle.com/competitions/deep-past-initiative-machine-translation/writeups/21th-byt5-base-gemini-augmentation-no-ensembl)
    - similar to first place.
- [22nd](https://www.kaggle.com/competitions/deep-past-initiative-machine-translation/writeups/23rd-place-cpt-grpo-no-sft)
    - has a nice what did/didn't work section
- [24th](https://www.kaggle.com/competitions/deep-past-initiative-machine-translation/writeups/25th-post-training-qwen2-5-32b-and-72b-with-gemi)
    - didn't do MBR
    - Basically a speedrun wow
    - [Qgentic-AI](https://github.com/bogoconic1/Qgentic-AI) would run experiments 
    - Qwen2.5-32B, Qwen2.5-72B and ByT5-Base
- [25th](https://www.kaggle.com/competitions/deep-past-initiative-machine-translation/writeups/26th-place-dpc-solution-contextual-byt5-large-e)
    - ByT5-large 
    - Didn't read the whole thing
- [27th](https://www.kaggle.com/competitions/deep-past-initiative-machine-translation/writeups/silver-medal-solution29th-byt5-xl-custom-cpt-wi)
    - ByT5-XL 
    - bf16
    - knowledge distrib
    - span corruption using akkad dictionary?
    - good what worked/didn't chart
- [30th](https://www.kaggle.com/competitions/deep-past-initiative-machine-translation/writeups/byt5-large-with-word-level-dapt)
    - ByT5-Large 
    - lotsa external data
    - mbr single and ensemble
- [32nd](https://www.kaggle.com/competitions/deep-past-initiative-machine-translation/writeups/32nd-byt5-base-fine-tuning-no-llms-no-synthet)
    - no synthetic, no llms, no ensembles
    - akkademia corpus pretrain
    - has "what didn't work section" called "Key Findings from Leaderboard Submissions"
    - "What Limited Us" is a great phrasing
- [47th](https://www.kaggle.com/competitions/deep-past-initiative-machine-translation/writeups/multi-phase-curriculum-training)
    - "to @phucthaiv02 for the PDF-extracted Akkadian translation dataset 🙏"
    - Joint Dropout
    - All phases: batch size 16, gradient accumulation 2 (effective 32), label smoothing 0.1, weight decay 0.01, cosine scheduler.
    - Phase 2 uses a 10× lower LR (1e-5) to prevent catastrophic forgetting of all the pretraining knowledge. Early stopping + load_best_model_at_end ensures we pick the best checkpoint.
    - Prefix for all inputs: "translate Akkadian to English: "
    - NO MBR
- [49th](https://www.kaggle.com/competitions/deep-past-initiative-machine-translation/writeups/2-stage-cv-cpt-and-mbr-decoding)
    - byt5-base 
    - Model soup
    - Single MBR
- [50th](https://www.kaggle.com/competitions/deep-past-initiative-machine-translation/writeups/56th-solution-byt5-base-tapt-ensemble)
    - "gratitude to @takamichitoda @vitorhugobarbedo for publishing outstanding notebooks and @phucthaiv02 for publishing akkadian_english_sentences_alignment_2 dataset in the hugging face."
    - four ByT5-base ensemble with MBR
    - 5 fold cross validation, with some fancy fold dividing?
    - translation data: Only keep ' " : . - for punctuation
    - LEARNING_RATE = 2e-4
        MAX_LENGTH = 512
        EPOCHS = 10

        label_smoothing_factor=0.1,
        lr_scheduler_type="linear"
        warmup_ratio=0.05,
- [59th](https://www.kaggle.com/competitions/deep-past-initiative-machine-translation/writeups/59th-place-silver-byt5-base-llm-data-augmenta)
    - ByT5-base 
    - 3-model cross-family MBR (Minimum Bayes Risk) ensemble
    - Data: ~10,700 sentence pairs (original train + OARE published texts + LLM-generated translations)
    - Has a VERY extensive what didn't work, and what they would do differently sections!!!!
        - What Didn't Work For Me

            Upgrading LLM from GPT-5.2 to GPT-5.4 for data generation: The newer model produced more fluent English, but this actually hurt translation quality. The ground truth translations in this competition are scholarly/literal (e.g., preserving Akkadian sentence structure, keeping technical terms), not fluent English. GPT-5.4's tendency toward natural-sounding prose introduced a style mismatch with the evaluation data. Sticking with GPT-5.2 (temperature 0.3) gave more literal, domain-appropriate outputs.
            PN (Proper Noun) Normalization: Implemented using OA Lexicon with fuzzy matching, but disabled in final submission — too risky for MBR scoring where small text changes could misalign candidate consensus. (Note: many top teams used this effectively — the issue was likely my integration with MBR, not the technique itself.)
            Translation Memory: Exact-match lookup from train.csv. Worked in isolation but conflicted with MBR ensemble. (Again, top teams successfully used TM — the conflict was specific to my MBR pipeline design.)
            Bidirectional training (Akk→En + En→Akk): Excluded in final configs — didn't improve quality. (Some top teams reported gains from this, possibly because they had cleaner data to begin with.)
            OOF Cascade Refinement: Trained a second-stage model on [SRC] + [DRAFT] → [REFINED] using 3-fold CV drafts. Interesting but didn't improve LB.

        - What I Would Do Differently (Lessons from Top Solutions)

            VLM-based PDF extraction pipeline: The #1 differentiator between Gold and Silver. Top teams used Gemini Pro / Qwen-VL to extract 20k-60k sentence pairs from academic PDFs (AKT series, etc.). I relied on OARE + LLM translation, which was lower quality.

            Scale up to ByT5-Large/XL: With enough clean data, larger models clearly win. CTranslate2 quantization (int8) makes XL feasible on T4 within the 9-hour kernel limit.

            Continual Pre-Training (CPT): Domain-adaptive pre-training with span corruption on raw Akkadian text before SFT. Almost all Gold teams did this; I skipped it entirely.

            Loss weighting by data quality: 1st place weighted high-quality data at 1.3-1.4x and LLM data at 0.5-0.7x. My source-prefix approach was a rough proxy but not as effective.

            Creative synthetic data: 3rd place generated 300k+ "vocabulary/grammar drills" from dictionaries. 5th place used back-translation. My LLM augmentation was straightforward translation — I should have been more creative about teaching the model Akkadian structure.
- [68th](https://www.kaggle.com/competitions/deep-past-initiative-machine-translation/writeups/75th-place-solution)
    - Just data cleaning
    - Ensemble of models? Used [public notebook](https://www.kaggle.com/code/waterjoe/lb-35-9-ensembling-post-processing-baseline) 
- [72nd](https://www.kaggle.com/competitions/deep-past-initiative-machine-translation/writeups/79th-place-solution)
    - ByT5-base and ByT5-large 
    - Iterative Improvement: We filtered samples from train.csv that achieved high scores based on the Geometric Mean of BLEU and chrF++
        - We then performed additional fine-tuning on this high-quality subset, which yielded an improvement.
    - 3. Ensembling & Inference Strategy
        To bridge the gap, we leveraged a bit of luck and extensive searching within the Kaggle "Models" and "Datasets" sections. We identified three high-performing public models and integrated them into our solution.
    - MBR Decoding: Crucially, we incorporated the official competition metric directly into our MBR decoding process to maximize our score.
    - num_beam_cands:      int = 8
        num_beams:           int = 8
        length_penalty:      float = 1.0
        repetition_penalty:  float = 1.1
        num_sample_per_temp: int = 8
        temperatures:        [0.70, 0.90]
        mbr_pool_cap:        int = 64 
    - don't overtune outputs to public leaderboard
    - final training on a subset of train.csv
- [89th](https://www.kaggle.com/competitions/deep-past-initiative-machine-translation/writeups/97th-place)
    - span corruption
    - no external data
    - project links included
        - span corruption
        - best model
    - Approaches that did not work:
        - Pseudo-labeling,
        - training a ByT5 model from scratch,
        - adding LSTM/GRU layers as a custom head on top of T5.
- [104th](https://www.kaggle.com/competitions/deep-past-initiative-machine-translation/writeups/one-shot-silver) 
    - solution is a weighted ensemble of 8 distinct pipelines, 4 using Helsinki model and 4 using byT5 base model.
    - Finetuned off of publicly available byt5 models to save time.
    - Pipeline [here](https://www.kaggle.com/competitions/deep-past-initiative-machine-translation/discussion/684249)
    - 2. Inference & Candidate Generation

        For each of the 8 models, inference is run using Beam Search with

            num_beams=4
            num_return_sequences=4.
            This generates exactly 4 translation candidates per model for every single input text.
            Total Pool: 8 models × 4 candidates = 32 candidate translations per row.
            Hardware Optimization: Inference scripts are run concurrently across 2 GPUs (CUDA_VISIBLE_DEVICES=0 and 1) using bash background processing to save time.
    - MBR for ensemble result
        - weight by model quality and competition metric.
        - remember to remove duplicates
    - Multitask learning (BT, etc)!
    - Regularized Dropout didn't get tested. 6th place had a regularized dropout implementation though.
    - Curriculum learning needs improvement !
    - final fine tuning on train.csv
    - wish they could have quanitized.
    - wish they didn't use train.csv
    - Qwen7B attempt couldn't be completed due to lack of time.
    - [Round-Trip RL](https://arxiv.org/pdf/2601.12535v1) with byT5 failed. But apparently could maybe have been improved.
- [105th](https://www.kaggle.com/competitions/deep-past-initiative-machine-translation/writeups/make-the-most-of-all-available-resources)
    - Very good documentation of experiments.
    - 3. Key Conclusions

        Quality over Quantity: ByT5 is extremely sensitive to data quality. LLM-generated translations act as noise and degrade performance. Only human/expert translations improve the score.
        Data Cleanliness is Critical: Hidden database artifacts (like 2.83333 instead of fractions) or unexpected languages (German/French) will immediately drop the score.
        Training Strategy: Unidirectional training (Akkadian → English) is required for augmented data to prevent reverse-direction interference.
        Perhaps The Best Combination: Train on train.csv + Step 3 (V2 Publications) + Step 4 (Cleaned Sentences) + Step 5 (Michel Corpus).
        For various reasons, I have not been able to make full use of all the data sets available.
        These are some datasets I have compiled myself; due to time constraints, I have only used two of them:augmented_step3v2_publications.csv
    - provides dataset.
- [160th](https://www.kaggle.com/competitions/deep-past-initiative-machine-translation/writeups/164th-place-solution-218-shake-up)
    - Data engineering with Gemini: 36,750 training pairs from scholarly PDFs. mT5-large 4-fold ensemble + MBR decoding.
    - "eos" token for end of sentence marking with concatenating akkadian sentences together after ocr pipeline makes sentences.
    - gemini to do data
    - Started with ByT5-base (byte-level, no OOV issues) but switched to mT5 for speed and vocabulary coverage. mT5's subword tokenizer produces ~4x shorter sequences than ByT5's byte-level encoding, enabling ensemble + MBR within Kaggle's 9-hour time limit.
    - Model 	mT5-large
        Optimizer 	AdamW, cosine LR 5e-5, 5% warmup
        Batch 	8
        Epochs 	20-40, early stopping patience 5
        Label smoothing 	0.1
        Precision 	BF16
        Validation 	series_group_kfold (5 folds, leave-series-out)
    - Validation strategy: series_group_kfold (5 folds, leave-series-out) -- a GroupKFold where each fold holds out an entire publication series (e.g., AKT 6a). This prevents series-level data leakage, since documents within the same series share vocabulary, formulaic expressions, and proper nouns.
    - MBR Decoding

        For each source, generated 10 candidates per model:

            4 beam search (num_beams=4, length_penalty=1.08)
            6 sampling (temperatures 0.60/0.80/1.05, 2 each, top_p=0.92)

        GeoMeanMBRSelector picks the candidate with highest average pairwise sqrt(BLEU * chrF++) against all others.
    - 4-Fold Ensemble + Char Constraint
        5-fold CV but only 4 folds used for ensemble -- 5 models × 10 candidates × ~4,000 sentences exceeded Kaggle's 9-hour time limit.
    - pre and post processing.






### takeaways:
- It's good to explore with a smaller model version, then try out a bigger version to see if it works better. not sure when this would be the case that it would work better though.
- both 8 and 12 (and maybe others) didn't use any resources which costed money. so its perfectly reasonable to make gold without shelling out money for api calls.
- For low data environments, its almost always a better use of your time to get more data and clean the data you have, than to try to do augmentation tricks with the data.
    - If you have all possible data, then do augmentation.
    - There were no posted solutions which only did data augmentation without at least grabbing all of the provided competition data from the PDFs.
        - I only used train.csv
- lower aaruond 100 place had some public model integrations
- many said dont use postprocessing. but some teams did use postprocessing
- many used mbr, but some did worse with mbr. Why?
- 


### Most shared ideas that maybe I ought to implement:
- MBR
- Ensembling
- Check any project links or cited public notebooks/posts
- An automated pdf extraction pipeline (use someone's premade one if it exists to save time!!!)
- the main thing I found helped in my experimenting was the sentence splice. So I wonder if that would help on top of someone else's solution
- 4th and 6th have extensive code I think, so they might be particularly great to look at, but also some of the lower ranked ones had some thorough code postings.
- I should not do the chatgpt pipelines because those cost money. especially first place costed quite a bit apparently? idk maybe do a small one. or the one that was free in 9th place or smth.