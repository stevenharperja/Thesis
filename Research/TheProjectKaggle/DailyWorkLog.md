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
