

it would be pretty cool to pipe the day's scheduled flashcards into an LLM before going through the cards. Ask it to make the best sentence it can out of all of the words, that way you get some reading material to link it to before/after you do your flashcards.
-   It just about works https://chatgpt.com/share/69316b62-6164-8007-bb0a-5c9b2505a381
What if instead of trying to make one big sentence, you tried to make a sentence for each word piped diri incorporate 1-2 other words

It looks like there is already a good amount of coverage for making flash cards for LLMs

Instead focus on that one idea of basket analysis for flashcard recommendation maybe? I don't think that has been explored much judging from a quick search result.
piping that directly into a dictionary could be good. as like "related words"
- this exists: https://relatedwords.org/


Could it be possible to compare a list of "known words" and a passage of text and select all the ones that aren't in the text, then have the llm make flashcards? actually that could just be done with traditional methods for the most part.




#### AAAA I WANT TO DO SOMETHING WITH MODEL COMPRESSION DAMMIT.
- Requirements:
MS with published research work in ML model optimization, post-training quantization, consideration of different datasets and different constrains (bit-accuracy, model size, latency and so on)
    Experience with popular ML frameworks, such as PyTorch and TensorFlow
    Experience with embedded systems and software SDK
    Startup mindset/experience

 

Experience in one or more of the following areas is considered a strong plus:

    Experience with popular light-weight ML models on edge inference
    Hands-on experiences with deploying/evaluating ML models on resource/power-limited computing platforms.
    Experience providing technical leadership and/or guidance to other engineers
    Hands-on experience on developing compiler libraries or tools
    Hands-on experience with driver development for ASIC/FPGA https://ewu.joinhandshake.com/jobs/10242604


#### Federated learning LLMs
I could do make it provide reading comprehension questions or rephrase flashcards. The benefit behind using federated learning to do it could be that it would work offline? users could provide tags for whether an output was good or not or something. Additionally it could be extended to making reading comprehension questions about local text that the user has access too, but might not want to share, or could reveal personal information about the user, so that would be another benefit of using federated learning here.
The downside would be that its working with a local model, and might not work very well, but I think it'd be fun to try and see how it goes.


### Free response questions autoencoder.
Maybe I could do some sort of autoencoder answer checker thing? as flashcard? they write a sentence then it gets fed into an autoencoder to check the vector distance or something rather than doing an exact match?

Maybe that could be the compression or federal learning attribute?

Could be an E-Learning website type thing. For grading quizzes and stuff. 
I've heard free response questions demonstrate learning better, 
so perhaps doing an autoencoder to automate this could make this more accessible for an e-learning perspective.

basically:
1. encode the true answer
2. encode the student answer
3. compare using a distance metric.
To train:
Make a lot of responses that are similar and train the vectors to be close. train ones which are wrong to be further away.
Maybe that would work idk.
Gosh it would probably be garbage actually since logic in LLMs is bad.
maybe it would work for checking grammar. frick idk.

