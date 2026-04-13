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
### Tokens in byt5
- No tokens, just bytes.
- Why this is good for Akkadian

# Related work
## Low Resource Languages
### Difficulties of low resources
- Not enough data.
### Data Augmentation
- Techniques
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
 