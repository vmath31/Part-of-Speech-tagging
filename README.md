### Naive Bayes
A part-of-speech tag is assigned to a word based on the number of times that word has occurred in the training set with a speech tag. Thus the tag is determined by the maximum probability of a speech tag given a word, represented by *argmax P(Si=si|W)* over all si.

### Hidden Markov Model and Viterbi
Based on the visualisation of the Bayes net, the HMM, our observed values are ‘words’ in the sentence and ‘tags’ are the latent variables. Each tag is dependent on the word in the sentence and, based on the Markov property, the tag prior in the sequence.  To represent these dependencies, transition *(Si | Si-1)* and emission probabilities *(Si |W)* are saved while training the train dataset. While testing, if any word, say w,  from the test data is not present in the training data, the emission probability of *Si=si|w* is assigned a very small value for the purpose of calculations, giving transitional probabilities a greater role to pay to determine the tag for W.
The Viterbi algorithm allows us to include the influence of more than just the immediately prior variable by saving the probability of assignment the previous tags to previous words in the sentence. These added dependencies make HMMs a better method for POS tagging indicated by as the sentences correctness is which almost always 10% greater than Naïve Bayes method.

### Gibbs Sampling and Max Marginal
Gibbs Sampling is a Monte Carlo Markov Chain that has helped attain relevant samples for attain probability of a tag given a word. The idea behind MCMC is to find the relevant samples from the posterior distribution. 
Every sentence is initialised with a random set of tags.  Relevant samples are found over several iterations, where the besides the tag for a word being sampled, all the other tags are kept constant. The (conditional) probability of a tag, at certain position in the sentence i is *P(S_i | S_i+1, S_i-1, W_i)* calculated on the following conditions:

1. First word of the sentence: ```P(S_i) * P(W_i|S_i) * P(S_i+1|S_i)```
2. Last word of the sentence:``` P(S_i|S_i-1) * P(W_i|S_i) * P(S_i|S_0), where S_0 is the tag assigned to the first word```
3. Any other position in sentence: ```P(S_i+1|S_i) * P(S_i|S_i-1) * P(W_i|S_i)```
4. When the length of sentence is only 1 word: ```P(W_i|S_i) * P(S_i)```

Sampling this way, allows the generated sample to follow a stationary distribution, returning the sought after posterior distribution in the end.
When the probability for each tag for one word is determined, it is normalised from which a random value and its associated tag is chosen with the help of its cumulative probabilities. When each word gets assigned a random tag, it is considered a sample. This does iteratively collects multiple samples, a data set from which we calculate maximum marginal of a word to find the most appropriate tag. However all the generated samples are not passed on as the final sample to calculate max marginal of a word to get the best possible results, Gibb's sampling does not given relevant samples in the initial iterations. This method gives better results that naïve Bayes and similar results to HMM and Viterbi.

### Assumptions:
1. Emission probabilities for words that do not occur in training dataset are assumed to have a low probability of occurring given tag
2. Transitions between tags that didn’t occur in the training dataset are given a low probability of occurring. 
3. Although Gibb’s sampling indicates keeping all other values constant besides the value being updated, not all other values have been used to calculate the probability of the tag-- only probabilities of previous tag, next tag and current word with respect to-be-sampled tag  is used. 

### Problems faced and their solutions
1. Time consuming: Reduced the number of iterations and burn-in samples, which gave a similar or less accurate result.
2. Underflow when calculating log of posterior: Instead of taking the log after calculating the posterior distribution, sum of logs was taken to deal with the underflow.


### Final Result

#### (in code) 1000 iterations and final sample of last 800
 ```
 So far scored 2000 sentences with 29442 words.
                  Words correct:     Sentences correct: 
 0. Ground truth:      100.00%              100.00%
 1. Simple:             91.76%               37.75%
 2. HMM:                93.57%               47.20%
 3. Complex:            93.50%               46.10%
```

#### 2000 iterations and final sample of last 1500
````
So far scored 2000 sentences with 29442 words.
                   Words correct:     Sentences correct: 
 0. Ground truth:      100.00%              100.00%
 1. Simple:             91.76%               37.75%
 2. HMM:                93.57%               47.20%
 3. Complex:            93.35%               45.05%
 (Takes 30+ min)
 ````
 #### 5000 iterations and final sample of last 4000
 ````
 So far scored 2000 sentences with 29442 words.
                   Words correct:     Sentences correct: 
   0. Ground truth:      100.00%              100.00%
         1. Simple:       91.76%               37.75%
            2. HMM:       93.57%               47.20%
        3. Complex:       93.44%               46.20%
````
