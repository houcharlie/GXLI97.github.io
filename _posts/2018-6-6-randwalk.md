---
layout: post
title: Notes on RAND-WALK - A Latent Variable Model Approach to Word Embeddings
---

Here are some notes for a paper I read recently, called [RAND-WALK: A Latent Variable Model Approach to Word Embeddings][1]. I've been reading some material on word embeddings recently, and this paper was especially interesting because it proposes a simple, intuitive generative model that explains why low-dimensional word embeddings "work" - i.e. give us the **King - Man + Woman = Queen** phenomenon that is always the first thing someone learns when they are introduced to word embeddings. The theory it proposes is also used to justify the popular models that are used in practice: the [pointwise mutual information][2], [word2vec][3], and [GloVe][4].

I will begin by summarizing the key points of this paper, and then diving into some proofs/details that I think are the most interesting.

A popular way to construct vector representations of words comes from a *distribution hypothesis*: "you shall know a word by the company it keeps". Given a large corpus of text, we can construct a matrix of (words, contexts). Contexts can be defined in many ways: for example as the "document" the word appears in, or as another word that is found in a "window" around the original word. For example, if we have a vocabulary of size $$n$$, we can construct a co-occurence matrix $$M \in \mathbb{R}^{n\times n}$$, where $$M_{ij} = $$ number of times word i and word j are found within distance $$k$$ of each other.

Usually, we apply some nonlinearity $$\phi$$ to $$M$$, and then do an singular value decomposition on $$\phi(M)$$, taking the top $$d$$ eigendirections as our $$d$$-dimensional embedding. Popular choices of $$\phi$$ include an elementwise $$\log$$ or the pointwise mutual information $$\text{PMI}(w, w') := \log\frac{p(w,w')}{p(w)p(w')}$$.

We also have the neural network language models such as word2vec, where our "word embedding" is simply the vector that the neural net represents the word as. [Levy and Goldberg, 2014][5] makes an attempt at drawing the connection between PMI and word2vec.

Some issues to consider:

* None of the generative models of language have been linked to PMI models.
* We don't have a good understanding of why the nonlinear PMI metric works - it's a "bit mysterious" as the paper says.
* We don't know why these low-dimensional embeddings have linear structure that allows us to solve analogies. As the paper points out, a low-dimensional embedding should have relatively large error that is bigger than the difference between the best solution and the second best (incorrect) solution. But in practice, we are able to get the best solution with high accuracy. 
* The explanation that low-dimensional embeddings work well because "smaller models generalize better" is fallacious, because we are not actually *training* our model to solve analogies. There's no reason why these models should be able to solve analogies, just like there's no reason they should be able to predict tomorrow's weather.

Key contributions of the paper:

* A simple generative model that gives us linear structure (what the paper calls "RELATIONS=LINES").
* Why low-dimensionality is important (spatial isotropic assumption).
* Theoretical justification for PMI, word2vec, and GloVe. (Connections to these mysterious methods!)

## The Generative Model
Here are the details of the model:

* Let $$n$$ be total number of words, $$d<n$$ be the size of our embedding space.
* Each word has a latent vector $$v_w\in\mathbb{R}^d$$ which we are trying to recover.
	* These word vectors are assumed to be distributed uniformly in space. 
	* They are generated as $$v = s * \widehat{v}$$. $$s\in\mathbb{R}$$ is scalar r.v. with $$\mathbb{E}(s) = \tau = \Theta(1)$$ and $$s \le \kappa$$ for some constants $$\tau$$ and $$\kappa$$. $$\widehat{v} \sim N(0, I_d)$$.
* There is also a time varying *discourse vector* $$c_t\in\mathbb{R}^d$$ which represents what is being discussed. It is time varying because at each time, $$c_{t+1} = c_t + \delta_t$$ for some small random $$\delta_t$$. This has the interpretation that our "conversation topic" is slowly changing in time. $$\delta_t$$ is upperbounded in L2 norm by some $$\frac{\epsilon_2}{\sqrt{d}}$$.
* Then at each time t, a word is generated as follows (log-linear word production model):

$$ P(w|c_t) \propto \exp(\left<v_w, c\right>).$$

* Of course, we need some scaling constant so that our probabilities sum to 1. We call this the *partition function*, and it's easy to see that it takes the form:

$$Z_c = \sum_w  \exp(\left< v_w, c\right>).$$

Some interesting properties arise from this model:

The general approach is to write out the co-currence probability $$p(w,w')$$ by integrating out the terms that involve conditioning on the contexts $$c, c'$$:

$$\begin{align}
p(w,w') &= \mathbb{E}_{c,c'}[P(w, w'|c,c')] \\
	&= \mathbb{E}_{c,c'}[P(w|c) P(w'|c')]
\end{align}$$

where the first equality is due to the law of total expectation, and the second due to the conditional independence.

However, the partition function $$Z_c$$ (which represents our scaling constant) is hard to work with. However, the paper shows that the partition function **concentrates** around a constant $$Z$$, which allows us to replace our $$Z_c$$ (which depends on the specific discourse vector) with a global constant.

Once this is in place, we can derive closed form expressions for word probabilities:


$$\begin{align}
\log p(w,w') &= \frac{||v_w + v_{w'}||_2^2}{2d} - 2\log Z \pm \epsilon \\
\log p(w) &= \frac{||v_w||_2^2}{2d}-\log Z \pm \epsilon \\
\text{PMI}(w,w') &= \frac{\left< v_w, v_{w'}\right>}{d} += O(\epsilon)
\end{align}$$

The last equation corresponds to our intuition that the PMI corresponds to an inner product between the two vectors (which we are trying to recover when we take the SVD).

## A New Training Objective

###Squared Norm
Proving the previous properties requires some mathematics which I currently don't 100% understand, but once we have it, a training objective can be derived with simple algebra.

We proceed by maximum likelihood estimation on the observed coocurence matrix $$X_{w,w'}$$:

$$
\begin{align}
l &= \log \left(\prod_{(w,w')} p(w,w')^{X_{w,w'}} \right) \\
&= \sum_{w,w'} X_{w,w'} \log p(w,w') \\
&= \sum_{w,w'} X_{w,w'} (\log{\frac{\widetilde{L}p(w,w')}{X_{w,w'}}} + \log \frac{X_{w,w'}}{\widetilde{L}})
\end{align}$$

here $$\widetilde{L} := \sum_{w,w'}X_{w,w'}$$ is sum of all the cooccurences. Then let $$\Delta_{w,w'} = \log{\frac{\widetilde{L}p(w,w')}{X_{w,w'}}}$$ (log of the ratio between expected count and observed count).

$$
\begin{align}
&= c+\sum_{w,w'} X_{w,w'} \Delta_{w,w'}
\end{align}$$

For some constant c that we don't care about because it doesn't depend on our word vectors ($$\Delta$$ does).

Instead of maximizing this directly, instead we will rewrite the expression $$\sum_{w,w'} X_{w,w'} \Delta_{w,w'}$$. To begin, let's look at $$\widetilde{L}$$:

$$
\begin{align}
\widetilde{L} &= \sum_{w,w'}\widetilde{L} p(w,w') \\
&= \sum_{w,w'} \exp(\log \widetilde{L} p(w,w')) \\
&= \sum_{w,w'} \exp(\Delta_{w,w'} + \log(X_{w,w'})) \\
&= \sum_{w,w'} X_{w,w'} \exp(\Delta_{w,w'}) \\
&= \sum_{w,w'} X_{w,w'} (1 + \Delta_{w,w'} + \Delta^2_{w,w'}/2 + O(|\Delta_{w,w'}|^3))
\end{align}$$

The paper makes the argument that the last term in the Taylor series is negligible, so we have:

$$
\begin{align}
\widetilde{L} &\approx \sum_{w,w'} X_{w,w'} (1 + \Delta_{w,w'} + \Delta^2_{w,w'}/2) \\
&=  \widetilde{L} + \sum_{w,w'} X_{w,w'} (\Delta_{w,w'} + \Delta^2_{w,w'}/2)
\end{align}
$$

So:

$$c-l \approx \sum_{w,w'} X_{w,w'} \Delta^2_{w,w'}$$

Maximizing $$l$$ corresponds to minimizing:

$$\begin{align}
\sum_{w,w'} X_{w,w'} \Delta^2_{w,w'} = \boxed{\sum_{w,w'} X_{w,w'}\left(\log X_{w,w'} - ||v_w+v_w'||_2^2 - C\right)^2}
\end{align}$$

### PMI

By directly considering the PMI equation above, we can also make a PMI objective by minimizing:

$$\boxed{ \sum_{w,w'} X_{w,w'} \left( \text{PMI}(w,w') - \left< v_w, v_w' \right> \right)^2.}
$$

Both objective functions can be optimized with AdaGrad.

In addition, there are very clear connections to GloVe and word2vec (continuous bag of words) in this model (Section 3).

## RELATIONS=LINES
Just to summarize this section, the paper considers the task of solving analogies. Previously we had stated that an issue with previous work is the fact that a low-dimensional embedding has a large approximation error that might confuse the top two solutions - however in practice we are usually able to get the correct solution. 

* GloVe starts with the *assumption* of linear structure and tries to build an objective from it, which is different from the approach of this paper where they start with a generative model and show that such a linear structure exists.
* Levy and Goldberg show that SGNS vectors satisfy this linear relationship, but their argument only holds for high-dimensional embeddings.

This paper shows that the generative model that was introduced has this linear structure, and the low-dimensionality of word vectors plays an important role. For every pair of words $$a, b$$ that have a relation R, $$v_a - v_b = \mu_r + \text{noise}$$. The dimension reduction can be viewed as a linear regression that when solved reduces the noise.

## Concluding Thoughts

This was an interesting paper to read, and fits nicely with other methods. I'm planning on running some of their sample code to reproduce the results.

[1]: https://arxiv.org/abs/1502.03520
[2]: https://dl.acm.org/citation.cfm?id=89095
[3]: https://arxiv.org/abs/1301.3781
[4]: https://nlp.stanford.edu/pubs/glove.pdf
[5]: https://papers.nips.cc/paper/5477-neural-word-embedding-as-implicit-matrix-factorization 
[6]: https://www.cs.toronto.edu/~amnih/papers/threenew.pdf
