2/3/2026
- Looked at the german pdfs. Decided not to do them.
- Consulted chatgpt for information on how to do mask fill with byt5 model (seq2seq models cant use huggingface fillmask dataloader)
    - read source from https://mbrenndoerfer.com/writing/span-corruption-t5-pretraining-objective
        - Notable takeaways:
            - Span Corruption is a good keyword. Also some code is provided, though no huggingface style code provided.
            - Turning on prefix language modelling (bidirectional attention for encoder) could be a good option.
            - 15% (from Myu value of 3) is a good option. Higher values will cause training time to be slower due to autoregression generation? but it slightly reduces input sequence read cost(time?) so ??
        - https://github.com/jkallini/mrt5 this might have practical code to implement it? //No
- Connected fill-in-blanks training file to github for ease of tracking.
- Searched for byt5 span corruption implementations, did not seem to find any.
    - I'll just have to adapt the code from the mbrenndoerfer article with the T5 [article](https://discuss.huggingface.co/t/how-can-i-pretrain-t5-model/168563?utm_source=chatgpt.com) I found earlier

2/4/2026
- Looked for ways to implement (or copy) a data collator for byt5
    - Read:
        - the transformers api https://huggingface.co/docs/transformers/main_classes/data_collator
        - https://towardsdatascience.com/data-collators-in-huggingface-a0c76db798d2/
    - After some thought, it occurs to me that using a mlm trainable model and comparing if mlm improves the training could be a decent indicator on whether building a dataloader is worth it? maybe? but I don't know if itd be accurate plus I kinda wanna do it anyway to put it on my resume?
    - Read:
        - https://discuss.huggingface.co/t/how-can-i-pretrain-t5-model/168563?utm_source=chatgpt.com
            - https://huggingface.co/docs/transformers/v4.34.0/model_doc/byt5
            - https://www.cs.princeton.edu/courses/archive/fall22/cos597G/lectures/lec03.pdf
                - Looks like good info.
                - https://www.cs.princeton.edu/courses/archive/fall22/cos597G/
                - To read in future: (likely wont get to it)
                    - (book) On the Opportunities and Risks of
Foundation Models https://arxiv.org/pdf/2108.07258
                    - (book) Speech and Language Processing https://web.stanford.edu/~jurafsky/slp3/
            - T5 pretraining reference. https://github.com/PiotrNawrot/nanoT5

2/5/2026
- Advisor meeting

2/6/2026 
- Found survey links for ML on Dead or low resource languages 

2/8/2026
- Looked through the book for techniques for dealing with having not many samples.
- Found more surveys specifically for translation for LRL 

2/12/2026
- More research
- Clean up bullet points.

2/13/2026
- Advisor meeting
- Research how to augment the data for transformers in code.

2/14/2026 
- Write sentence splice function and train model with the augmented data. (10 epochs)
- Read about how word2vec algorithm works.

2/15/2026
- Added random split and reran the training with 20 epochs instead of 10.
    - Returned a (validation) loss of about 0.3 (about 50% of previous, very good.), with a chrf of about 4 (about 80% of previous, not good. we want this to go up).

2/16/2026
- Ran the model for 10 more epochs to see if I could squeeze a higher chrf since it looked like just maybe it'd still go up.
    - Notably since I'm rerunning the notebook, it means I'm using different augmented data on this training round. Since I only (know how to?) initialize the data before the training starts.

2/17/2026
- Submitted model. obtained 26.4 score compared to 26.6 baseline. So the sentence splicing seems to have had little effect positively or negatively. And took about 1.5 times as long to train. 

2/18/2026
- Briefly compared qualitatively the sentences that both models models produced. They both look fine on the small scale but both look like nonsense sentences.
- Reran the sentence splice with 8 beams instead of 4. Notably I don't think I've used 8 beams for the baseline yet to compare with.
    - 8 beams for the sentence spliced achieved 26.6, which is the same as the baseline with 4 beams.
    - Apparently I lost the baseline model. So I will likely have to retrain it again if I can't find it.
- Found various papers on bilingual word embeddings.
    - particularly interested in this one: [Bilingual word embedding fusion for robust unsupervised bilingual lexicon induction](https://www.sciencedirect.com/science/article/pii/S1566253523001343) 

2/19/2026
- Added TODO for how to compare results of the models better.
- Added more details for how the sentence splice performed.
    - I think I might be able to train the sentence splice one longer and see if I could get an even higher chrf score. It doesn't look like it's really overfitted much yet.
- Read introduction section of [Translating Akkadian to English with neural machine translation](https://academic.oup.com/pnasnexus/article/2/5/pgad096/7147349?login=true#412513286) paper.
- Advisor meeting
- Ran sentence split for another 10 epochs (total 20+10+10=40)

2/20/2026
- Ran sentence split for another 10 epochs (total 20+10+10+10=50)
    - CHRF value is still going up???
- Evaluated the 40 epoch one from yesterday with
    - 4 beams: 30.4
    - 8 beams: 30.4
    - 16 beams:30.4
    - this beats the 26.6 baseline i had
- eval'd 50 epoch, it had 30.4 for 4 beams

2/22/2026
- Made code for word2vec.

2/23/2026 
- Trained word2vec on akkadian.
- Started training a backwards EN -> AKKAD translator for 20 epochs using sentence split to see if that can make more data, and if it can help to make a forward translator later.
    - Plans:
        - After it is done training for 20 epochs I want to use it to make more data. Notably we should *not* train it very well, since it will be used on our same training data. 
            - Maybe I should have only done 10 epochs, or did a different split. Oh well.
        - I want to try using that data to train a fresh byt5. Ideally I'd like to try it on the baseline, the sentence split, and the backtranslator, but that is a lot of time that may not be worth it right now.
- Read over the backtranslation [paper](https://aclanthology.org/P16-1009.pdf) referenced by the [big paper](https://dl.acm.org/doi/full/10.1145/3524300#sec-3) I had, They used a 1:1 synthetic to real ratio it looks like.

2/25/2026
- Read through the rest of [Bilingual word embedding fusion for robust unsupervised bilingual lexicon induction](https://www.sciencedirect.com/science/article/pii/S1566253523001343#b31)
- Looked over the vecmap github page and others to see if there are more recent alternatives.
- Add TODO in word2vec notebook.
- Interestingly, numbers are the highest rate word in the dataset.

2/26/2026
- Advisor meeting

2/27/2026
- Made EN word2vec mappings.
- Run VecMap using default "Unsupervised" setting
- Test vecmap and rerun using "Identical" setting to make sure that the shared "THISISANUMBER" word is mostly aligned. 
    - They have cosine similarity of 0.68, Meaning that they are at a 47 degree angle from each other, which guarantees that they are in the same quadrant in 10D space.
    - Other words like šalimaššur and ennamaššur have even less cosine similarity at 0.26 (75 degrees) and 0.58 (54 degrees)
    - These degrees are not the worst, As long as they are less than 90 degrees for all major cases it should be usable. But if they could be within 30 degrees that would be preferable.
    - These similarities are subject both to inaccuracy caused by vecmap, and by inaccuracy from low data given to gensim.

3/4/2026
- Ran the backtranslation model to create a csv file.

3/5/2026
- Reran the backtranslation model after fixing a KeyError.
- Read over various Kaggle discussions on the competition.
    - This is useful: [Insights from the Akkademia Codebase & PNAS Paper for the Deep Past Challenge](https://www.kaggle.com/competitions/deep-past-initiative-machine-translation/discussion/673904)
        - [BPE size](https://www.rws.com/language-weaver/blog/issue-121-finding-the-optimal-vocabulary-size-for-neural-machine-translation/) which is [Byte Pair Encoding.](https://en.wikipedia.org/wiki/Byte-pair_encoding)
    - [Current state of the leaderboard as of 25 ish days ago?](https://www.kaggle.com/competitions/deep-past-initiative-machine-translation/discussion/672511)
- Advisor meeting
- Started training a fresh model on the backtranslation augmented data. (the backtranslation data was generated using the split sentence training for 20 epochs btw!)
- Reran vecmap with 15 dimensional vectors instead of 10 to allow more distinction between vectors. Didn't fix it that much but it should be better.
- ran the backtranslation-augmented model
    - got 29.1 on 4 beams.
    - got 29.1 on 8 beams.
- trained the fresh model on the backtranslation augmented data for another 10 epochs.
- Tried to make the synonym replacement train file. Ran into errors.

3/8/2026
- Ran inference with 30 epochs of backtranslation fresh. got 29.5 with 4 beams.
- Trained the fresh model on the backtr. for another 10 epochs (30->40)
- Ran it, got 28.8
- Trained a fresh model on the basic train/test but substitute numbers, and formatted it exactly like the word2vec words I used.
- Ran the substitute numbers model, got 14.9.
    - Realized that the test csv doesnt actually have any (roman numberal transcribed) numbers in it. So it wouldn't help anyway???

3/11/2026
- Looked over the gathered techniques page and marked a bunch as TODO.

3/12/2026
- trained synonym (mispelled as simile) model for 20 epochs.
- implemented the mask-fill training file.
- TODO: read byt5 paper: https://arxiv.org/pdf/2105.13626
- ran the mask-fill (pre)-training (EN&AKK) for 20 epochs.
    - NOTE! I did NOT split the training data for use in training later. This will invalidate the results I get from the training set. I SHOULD have set it to run on the untranslated dataframe provided by the competition. perhaps it will be interesting to compare the results of that with the current one I am running.
        - I have now prepared another file to run without it, but I can only run two sessions at once so I must wait an hour or two. for similes to finish.
- Synonym finished (20 epoch) with validation loss of 0.4ish, it looks like it still has room to train (note that it is stochastic like the sentence swap was).
- Infer synonym inference with 20 epoch 4 beam, got: 15.7, I think it needs more time to cook.
- Train the EN&AKK pretrained for main training for 20 epochs.
- Train Synonym 10 more (20+10)
- make akkadian only training set which includes stuff from entries which had gaps (split on gaps) and include only 25 char length or higher.
- Infer en/akk 20-20 epochs. it got 27.8 // This makes me think that combining with sentence split would be a good idea.
- Infer synonyms 20+10=30 epoch. It got: 15.9
- Train mask-fill with akkadian only (from non-training set). on sequences above 25 characters length. for 20 epochs. TODO consider making a for loop to redo after every epoch!!!! Please!!!

3/13/2026
- Train akk-only pretrained 20-20 train epochs.
- Infer akk-only pretrained 20-20 train epochs. result was 26.7. I feel like this isn't conclusively bad though. I wanna try again since the other one did so well.
- Modified akk-only pretrain code to do the randomization after each epoch.
- pretrain the akk-only for 20 more epochs (any more and quota will be too full unfortunately). //Will it go over? idk
    - Once it fills again tomorrow, I'll check the validation loss and see about pretraining it for longer. (training for longer is fine as long as it makes a better translator in the end.)
    - Afterwards, I can try training it generically, or with the sentence split, or other techniques I've used. I need to make the sentence split also shuffle every epoch so its better.
- Train the akk-only (20+1*20) for 20 epochs as regular training 
- Infer the above^, got: 26.5
    - Absolutely no improvement huh. why did the akk/en go up slightly? was it just a fluke? are the examples in akkadian too small?

3/17?/2026
- Retrain the Akk-only pretrain with mean span of 20 instead of 3 because the byt5 paper said they used 20 bytes. Also take away everything in that pretraining data less than 100 characters long.

3/19/2026
- Advisor meeting

3/25/2026
- Read over the 1st,3rd, and 6th and 7th place solutions
- "perhaps I should write my own write up on Kaggle while it's still fresh?"

3/26/2026
- Advisor meeting
- Read over the 5,8,10,11,12 place solutions

at some point I went over all the rest of the solutions before 4/02

4/02/2026 
- Advisor meeting

4/06/2026
- Begin bullet point draft for thesis.

4/09/2026
- Advisor meeting

4/10/2026
- Imported dataset from second place and made it ready for training.
    - Projected to take 40 hours on byt5-small with P100.
    - second place uses byt5-large
4/11
- ran it, exceeded kaggle time limit

4/12
- ran just one epoch and eval'd it, got 38 ish with byt5-small
- made plans
4/13
- ran one more epoch.
4/14
- ran one more epoch
4/15
- Filled out form to graduate
- filled out 3rd committee memberform
- Copied techniques (list of interesting ones) to draft.md