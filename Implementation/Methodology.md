1. Create a model with randomly initialized weights, or better yet, find a model online to fine-tune.
    - Models to consider:
        - BERT? 
        - LLAMA
        - GPT-2
            - https://github.com/openai/gpt-2
            - https://huggingface.co/openai-community/gpt2 May not be correct.
            - Looks like its 6 GB so maybe itll fit on the lab computer. I doubt it would train though.
        - RoBERTa
        - DeBERTa
    - Benchmarks:
        - GLUE
            - applies to BERT models
            - 
2. Have 2 copies of the model, one as the control, one as the original.
3. Train both of them on the same data, in the same order.
    - Consider using LoRA as a baseline so that training goes faster? idk. might get too complicated, perhaps better to just have a small model.
4. Compare performance.
5. Perform low rank factorization techniques
    - Which ones?
    - Just SVD
    - How do I make the model work afterwards?
        - Use this: https://docs.pytorch.org/docs/stable/generated/torch.svd.html
        - If the SVD process goes too slow, then find an optimized algorithm.
    - graph the frobenius norm error for each layer similar to https://web.stanford.edu/~pilanci/papers/lplr.pdf#appendix.J
6. Compare performance.



Ingredients necessary:
- A pre-trained model small enough to fit on my system.     More than one if possible, but for MVP just do one.
    - Try BERT?
    - Check the ML server RAM size.
- A compatible dataset for that model.                      More than one if possible, but for MVP just do one.
    - EWU Library guide page has that dataset search with google thing.
- Low rank factorization techniques to try.                 More than one if possible, but for MVP just do one.
    - Simple SVD technique?




//Before all of this, read each Low Rank Factorization technique from model compression surveys