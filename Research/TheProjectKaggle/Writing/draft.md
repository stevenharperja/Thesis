# Title Page
 Analysis of Neural Machine Translation Techniques for Translating Old Akkadian to English
# Abstract
Old Akkadian was an East Semitic language spoken in the Middle East about 4000 years ago. Only written records remain of the original language. A data science competition was held on kaggle.com to incentivize the improvement of techniques to translate this language to English. Thereby reducing the burden on the small number of people in the world who can decipher Old Akkadian. In this paper I explain my work during the competition, and analyze the techniques the most successful contestants used to translate this ancient, low-resource language.
# Acknowledgments
- Deep Past Initiative
- Contestants involved in the competition
- All those who have helped in the history of deciphering Akkadian.
- My advisor
- People whose stuff I cited? maybe?

# Table of Contents
 
# List of Figures
 
# List of Tables
 
# Introduction
## Kaggle Competition


The Deep Past Challenge was a Kaggle competition focused on the translation of Old Assyrian cuneiform tablets using neural machine translation techniques. The competition addressed one of the largest untranslated archives of the ancient world: thousands of clay tablets produced nearly four thousand years ago by Assyrian merchants operating trade networks between Mesopotamia and Anatolia. These tablets documented everyday and commercial activities, including debts, trade transactions, legal agreements, and family correspondence. Although approximately twenty-three thousand tablets survived, only about half had been translated due to the limited number of specialists capable of reading Old Assyrian Akkadian.

The competition framed this historical and linguistic challenge as a modern machine learning problem. Participants were tasked with developing models capable of translating transliterated Akkadian text into English. Old Assyrian Akkadian presented significant difficulties for standard translation systems because it is both a low-resource and morphologically complex language, where single words can encode information that requires multiple words in English. As a result, conventional neural machine translation architectures designed for modern, high-resource languages often performed poorly in this setting.

Beyond its immediate objective of translating ancient texts, the competition highlighted broader challenges in natural language processing for endangered, historical, and underrepresented languages. By encouraging the development of translation systems for Akkadian, the competition contributed to ongoing research into low-resource language modeling and demonstrated how machine learning approaches can support the preservation and interpretation of cultural heritage materials. // paraphrased from from https://www.kaggle.com/competitions/deep-past-initiative-machine-translation

## Translating Old Akkadian to English

# Literature Review  / Related work / Background
My literature review consisted mostly of papers about doing machine translation for low resource languages. These primarily covered techniques for data augmentation. So that is what I focused my efforts on.

Prior to this competition, there was not that many NML papers on Old Akkadian, but now there are dozens because of the competition. so in the later part of my thesis there was an additional, large amount of reading to cover. Most of these covered what worked/didn't work for them. And methods of processing massive amounts of pdf data was a consistent theme.

notable sources:
- ["A Visual Survey of Data Augmentation in NLP" ](https://amitness.com/posts/data-augmentation-for-nlp)
- [Low-resource Neural Machine Translation: Methods and Trends](https://dl.acm.org/doi/full/10.1145/3524300#sec-3)
- [Translating Akkadian to English with neural machine translation](https://academic.oup.com/pnasnexus/article/2/5/pgad096/7147349?login=true#412513286)
# Background
## Old Akkadian
### The time period, and archeology
- Use wikipedia a lot for this section. https://en.wikipedia.org/wiki/Akkadian_language
- 2334–2154 BC. Give a rundown of how it would have been in the Akkadian place.
- Agriculture was well established, cities, etc.  
- Show a map
- Show a tree of related languages.
- written in cuneiform, also used in other scripts in the geographical area
- heavy influence from sumerian
- oldest record of any Indo-European language (wikipedia source needed)
- Semitic Language.
    - East Semitic
- SOV
    - most other semitic have VSO or SVO
- has two unique prepositions that are pronounced differently in the other semitic languages
- mix of logograms and syllabic script, with some markers for divine names. Mainly syllabic? I'm not sure if "OLD" akkadian was syllabic though since wikipedia doesn't specify the times.
- Old Akkadian is older than Old Assyrian
    - Akkadian is divided into several varieties based on geography and historical period:[22]

    Old Akkadian, 2500–1950 BC
    Old Babylonian and Old Assyrian, 1950–1530 BC
    Middle Babylonian and Middle Assyrian, 1530–1000 BC
    Neo-Babylonian and Neo-Assyrian, 1000–600 BC
    Late Babylonian, 600 BC–100 AD
- "Old Akkadian, which was used until the end of the 3rd millennium BC, differed from both Babylonian and Assyrian, and was displaced by these dialects."
- For context, agriculture had already been going on for 2000-6000+ years. https://en.wikipedia.org/wiki/Neolithic_in_the_Near_East
- Include an example image of cuneiform to transliteration to english from one of the pdfs
- 
### The organization running the program
// not necessary?
### history of decipherment of akkadian

https://www.ebl.lmu.de/about/library

### Contents of the data
- trades, politics, etc.? (How do I analyze this?)
    - feed a section of them through an llm?
    - chatgpt says: "The overall dataset is heavily skewed toward economic and bureaucratic communication, with a smaller portion of personal correspondence."

    ["the competition tests Old Assyrian merchant correspondence exclusively."](https://www.kaggle.com/competitions/deep-past-initiative-machine-translation/writeups/9th-place-solution-for-the-deep-past-challenge-c)
## Byt5
### Transformer architecture
- Grab some stuff and cite Illustrated Transformer

For this competition, the Byt5 model was the most widely used. It is based on the transformer architecture which revolutionized natural language processing and was invented back in 2017 in the ["Attention is all you need"](https://arxiv.org/abs/1706.03762) paper. Then later the T5 architecture was invented for sequence to sequence processing, and then byt5 was made in a similar style.


### Tokens in byt5
- No tokens, just bytes.
- Why this is good for Akkadian

Instead of using tokens like most transformer models, Byt5 does not create tokens, but instead uses unicode raw bytes directly. This allows it to better deal with noisy data that has characters inserted, and also to better deal with data that uses unusual symbols, or has unusual letter sequences. This quality is very useful for Akkadian due to its high usage of accented characters. This allows the model to train faster, since a token model does not need to be trained.

# Related work
## Low Resource Languages
### Difficulties of low resources
- Not enough data.
The primary difficulty of working with low resource languages is that there is often not enough data to train a model with brute force by throwing more and more data at it. Instead it can be helpful to get clever and create fake, but similar data through a techniques broadly called "Data Augmentation" or "Data Imputation" to give the model more variance in what it sees and prevent overfitting to the small training set. Additionally, introducing noise into the data can also do a similar effect, as this is similar to what data augmentation is.

The best data augmentation techniques introduce new, accurate data, that has relevant new information not included in the base dataset. For sentences, this could be swapping out words with other grammatically correct ones.
### Data Augmentation
- Techniques
There are many different kinds of data augmentation techniques. 
Here is a list of ones I found while researching:
- Tried:
    - Fill-mask/seq break thing? TODO
        - Akk only
            - You add spans of blankness and have the model guess what must go into those blank sections.  This trains the model to better recognize common patterns of words in the text, so the intuition is that it will help with grammar and with familiarizing the model to the domain space. This is a common pre-training technique used across NLP. It would be useful here because it could make use of untranslated akkadian text as well, without requiring an english translation to have been made.
        - En-Akk
            - Taking the english and akkadian translation pair and considering it as one data point to put gaps into.
    - Instance Crossover Augmentation
        - You take two sentences and choose a point to cut each sentence off, cutting it into a A/B pair and C/D pair. Then you the later half of the sentences to create a A/D and C/B set of sentences. This roughly preserves grammatical structure while adding noise that the model must figure out. For a translation, you can do roughly the same thing just doing it for both the akkadian and english sides.
    - (Reverse) backtranslation
        - You translate backwards, from akkadian to english, then you translate that fake akkadian back to english, and you've now got a set of akkadian/english pairs which are both fake. This allows you to add noise to your data and can broaden the data the model will fit properly to.
    - Unsupervised Bilingual Word Embedding
        - You create word embeddings to be able to categorize the words and find their synonyms, or find words which fit similarly into a grammar structure. Then you can swap out similar words to create more training data that is still roughly correct. You can use libraries like Gensim to accomplish this. For a bilingual translation, vecmap is a popular technique to connect word embeddings from different languages together, so that you can find words which have the most similarity to each other across the languages. A dictionary would be a better approach, but word embeddings are easier to automate.
- Didn't try:
    - Sentence Reordering
    - (Forward and back) backtranslation
    - Random Noise Injection
        - Unigram Noising, Based on frequency distribution of a word
        - Blank noising, replacing words with "_" or some other thing.
        - Sentence Shuffling
        - Random Insert
        - Random Swap
        - Random Delete
    - Unsupervised Data Augmentation (TODO see book)
    - Uncertainty-Aware Self Training (TODO see book)


### Recommended search terms
- 
## Past papers on Akkadian
- [Translating Akkadian to English with neural machine translation](https://academic.oup.com/pnasnexus/article/2/5/pgad096/7147349?login=true#412513286/)
- Huggingface dataset  // TODO grab this
- TODO See if there are any other citations from the kaggle peeps
## The results of the kaggle competition
Almost every top placing competitor made their own dataset from akkadian research data that was outside of the ones provided in the competition. Many made a pipeline using OCR for pdfs, and utilizing LLMs for processing that data further, with some even making translations with LLMs.
A lot of the top solutions also did extensive processing to clean the data too. And their final solutions were all about having good data, rather than fancy model tricks.
This shows that in a low resource data environment, it is always best to get more data. Many of the competitors also did data augmentation techniques, but not all of them.
# Methodology
## My experimentation during the competition
### Techniques
- Fine-tuned arabic to akkadian model
- Fine-tuned arabic to akkadian model with frozen decoder
- Byt5 base comparison
- Dictionary+train_df trained model
- Sentence splice model (splice two different translation pairs together at a random point)
- backtranslation+train_df trained model
- Number substitution model
- Synonym data augmented model
- Pretrain "fill in the gaps" before regular training
### Difficulties and considerations
- Decided not to do a data pipeline on account of the large workload and specific applicability to this data.
    - OCR, etc.
- Focused on NLP data augmentation techniques because they seemed broadly applicable and were new and interesting to me.


# Results / Experimentation
## My experimentation during the competition, And results
### Techniques and results
- Fine-tuned arabic to akkadian model
- Fine-tuned arabic to akkadian model with frozen decoder
- Byt5 base comparison
- Dictionary+train_df trained model
- Sentence splice model (splice two different translation pairs together at a random point)
- backtranslation+train_df trained model
- Number substitution model
- Synonym data augmented model
- Pretrain "fill in the gaps" before regular training

## Kaggle 
### The journey of the competition
- Public and popularly cited code and Discussions
- Copying models, and baselines derived from them.
- Messages to/from the competition hosts
    - Formatting was a focus.
### The aftermath of the competition
- Overview of 1st through 101st writeups
- A focused overview of 1st through 10th
## Post competition 
### Analysis
- 7 ish categories of techniques used
    - Data Extraction
        - OCR
            - Most top teams extracted extra data from the pdfs provided from the competition, or went and found akkadian data from the akkadian respositories online. This would be difficult for me to replicate in a timely manner, so for my post-competition experiments, I decided to instead use data from teams who provided their extracted data rather than recreate it on my own.
    - Data Augmentation
        -  LLM rule based augmentation
            3rd place, (who else?)
        - Pseudo labelling (knowledge distillation)
        - Back translation
    - Training technique
        - Knowledge distillation
        - Multi language augmentation
        - Projected Gradient Descent
        - Backtranslation (5)
        - Exponential Moving Average (EMA)
        - Fine tuning to fit the train.csv after using their own data
        - Applying a penalty to the length
        - LoRA (how? 12)
        - Random Drop for model diversity (6)
        - Translation memory (9)
    - Pre processing
        - Repairing OCR
        - Matching guidelines shared by the host (15)
            - text normalization (2)
        - 100-400 length to match competition(11)

    - Post processing
        - MBR reranking
            - Many people used several different evaluation metrics beyond what the kaggle competition scored on. 
                - based on an [open source solution](https://www.kaggle.com/code/ngyzly/better-candidate-diversity-on-public-model?scriptVersionId=302266718) someone published during the competition 
                - metrics were: 
                    - chrF++
                    - BLEU
                    - token-level Jaccard similarity
                    - length prior
    - Model type
        - Large models worked for most teams.
            - Requires external resources so I did not replicate it.
    - Ensemble diversification
        - Ensembling
        - Multiple models
            - models trained on different data subsets
            - different model types
            - different model sizes
        - MBR again to glue it all together.

### Experimentation using that analysis
- TODO
- Used the 2nd place data and just trained a byt5-small model directly on it. It seems the data works well, getting over 38 on public and 40 on private score.
- Used the 8th place data to train a byt5 small model. It didn't work well. I think I mishandled the data
- I tried to use byt5-base which has more parameters than byt5-small, but it wouldn't fit on the gpu I had access to.
- //Didn't try yet
    - MBR
    - EMA?
    - Random drop
    - 
# Discussion

# Conclusion 

## Takeaways
### From competition:  
- It is better to get more data than to try to do data augmentation on inaccurate data.
- It's good to explore with a smaller model version, then try out a bigger version to see if it works better. not sure when this would be the case that it would work better though.
- Postprocessing on a model's output can overfit to your validation set.
- If you can create a book-work workflow that could be done by hand by a non-expert, even if it sounds impossibly slow, you can just automate that workflow with an LLM. 
    - provide any relevant knowledge necessary to do the job to the LLM. Like json exerpts from a book
    - An example could be a you could give someone. "here's these grammar rules for akkadian that are relevant to the following passage, here's every relevant definition,  give your best guess to try to translate this."
    - Primarily this thought comes from [3rd place](https://www.kaggle.com/competitions/deep-past-initiative-machine-translation/writeups/3rd-synthetic-data-to-teach-oa-fundamentals) 

# Future work
- These techniques can be used on other low resource languages or old languages which are hard to decipher.


# References
- Abdulla, F., Agarwal, R., Anderson, A., Barjamovic, G., Lassen, A., Ryan Holbrook, and María Cruz. Deep Past Challenge Translate Akkadian to English. https://kaggle.com/competitions/deep-past-initiative-machine-translation, 2025. Kaggle.

# Vita
## Author: Steven James Harper
Place of Birth: Montana ???//Why do I need this???
## Undergraduate Schools Attended: 
- Columbia Basin College
- Eastern Washington University
## Degrees Awarded: 
- Associates ??? 2020 Columbia Basin College
- Bachelor of Science in Computer Science 2024 Eastern Washington University
## Honors and Awards: 
- Graduate Assistantship, Computer Science and Electrical Engineering Department 2024-2026 Eastern Washington University
## Professional Experience
- Internship, Pacific Northwest National Laboratory, Richland, Washington, 2022

# Appendices
 