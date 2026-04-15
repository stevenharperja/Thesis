# Title Page
 
# Abstract

# Acknowledgments
 
# Table of Contents
 
# List of Figures
 
# List of Tables
 
# Introduction
## Kaggle Competition
## Translating Old Akkadian to English

# Literature Review  / Related work / Background
 
# Background
## Old Akkadian
### The time period, and archeology
- Use wikipedia a lot for this section.
- 2000 BCE. Give a rundown of how it would have been in the Akkadian place.
- Agriculture was well established, cities, etc.  
- Show a map
- Show a tree of related languages.
### Contents of the data
- trades, politics, etc.? (How do I analyze this?)
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
        - En-Akk
    - Instance Crossover Augmentation
    - (Reverse) backtranslation
    - Unsupervised Bilingual Word Embedding
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
## Post competition experimentation
- TODO
# Discussion

# Conclusion 

## Takeaways
### From competition:  
- It is better to get more data than to try to do data augmentation on inaccurate data.
- It's good to explore with a smaller model version, then try out a bigger version to see if it works better. not sure when this would be the case that it would work better though.
- Postprocessing on a model's output can overfit to your validation set.

# References
 
# Appendices
 