2/3/2026
- Looked at the german pdfs. Decided not to do them.
- Consulted chatgpt for information on how to do mask fill with byt5 model (seq2seq models cant use huggingface fillmask dataloader)
    - read source from https://mbrenndoerfer.com/writing/span-corruption-t5-pretraining-objective
        - Notable takeaways:
            - Span Corruption is a good keyword. Also some code is provided, though I don't know yet if they have huggingface style code provided.
            - Turning on prefix language modelling (bidirectional attention for encoder) could be a good option.
            - 15% (from Myu value of 3) is a good option. Higher values will cause training time to be slower due to autoregression generation? but it slightly reduces input sequence read cost(time?) so ??
        - Toread: https://mbrenndoerfer.com/writing/t5-pretraining-span-corruption-denoising-objectives 
- Connected fill-in-blanks training file to github for ease of tracking.
