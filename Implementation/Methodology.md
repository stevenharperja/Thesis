1. Create a model with randomly initialized weights, or better yet, find a model online to fine-tune.
2. Have 2 copies of the model, one as the control, one as the original.
3. Train both of them on the same data, in the same order.
4. Compare performance.
5. Perform low rank factorization techniques
    - Which ones?
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