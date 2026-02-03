2/3/2026
- Looked at the german pdfs. Decided not to do them.
- Consulted chatgpt for information on how to do mask fill with byt5 model (seq2seq models cant use huggingface fillmask dataloader)
    - read source from https://mbrenndoerfer.com/writing/span-corruption-t5-pretraining-objective
        - Notable takeaways:
            - Span Corruption is a good keyword. Also some code is provided, though no huggingface style code provided.
            - Turning on prefix language modelling (bidirectional attention for encoder) could be a good option.
            - 15% (from Myu value of 3) is a good option. Higher values will cause training time to be slower due to autoregression generation? but it slightly reduces input sequence read cost(time?) so ??
        - Toread: https://mbrenndoerfer.com/writing/t5-pretraining-span-corruption-denoising-objectives 
        - https://github.com/jkallini/mrt5 this might have practical code to implement it?
- Connected fill-in-blanks training file to github for ease of tracking.
- Searched for byt5 span corruption implementations, did not seem to find any.
    - I'll just have to adapt the code from the mbrenndoerfer article with the T5 [article](https://discuss.huggingface.co/t/how-can-i-pretrain-t5-model/168563?utm_source=chatgpt.com) I found earlier
    
