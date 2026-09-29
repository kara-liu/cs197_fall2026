# Project 3 & 4: Representation Learning

**PLEASE DO NOT SHARE THIS MATERIAL BEYOND CS197 AT STANFORD UNIVERSITY**

Machine learning (ML) has achieved major successes on text and images, supported by architectures designed for those data types. Tabular data remains both extremely common and comparatively challenging: electronic health records, surveys, census data, financial records, and many scientific datasets are naturally represented as tables. These datasets often mix continuous, binary, and categorical variables, contain missing values, and include features with very different scales, distributions, frequencies, and relationships with one another.

Separately, the field of **representation learning** asks if we can transform a complex, high-dimensional observation (e.g., an image, large text response, or tabular datapoint) into a smaller latent representation $^{[0]}$ that still captures useful information. Unlike a model trained only to predict one outcome, a useful representation can potentially support many downstream tasks.

In healthcare, for example, we might want a compact patient representation that summarizes information from demographics, laboratory measurements, diagnoses, medications, and vital signs. The same representation could then be used for tasks such as disease risk prediction or identifying similar patients.

The projects are as follows, with the hyperlinks attached:

- [Project 3 (Intermediate Level  🧩): Evaluating Numerical Preprocessing Methods for Tabular Representation Learning](#proj3)
- [Project 4 (Advanced Level 🌀): Information Retention in Tabular Representation Learning ](#proj4)

$^{[0]}$ The terms *latent*, *embedding*, and *representation* are used often interchangeably to denote a multivariate, often compressed, vector $z = f(x)$ of an input $x$, for some function $f$. Some works may also use the term *embedding* to refer to the output of preprocessing an input, i.e., a numerical embedding will be input into a model. 


## <a id='vae'></a> A Brief Intro Into VAEs

I recommend consulting the blog post by [Lilian Weng here.](https://lilianweng.github.io/posts/2018-08-12-vae/)

Variational autoencoders (VAEs) are generative models that learn a latent representation of high-dimensional data. Given an observation $x \in \mathbb{R}^d$, a VAE represents it using a latent variable $z \in \mathbb{R}^k$, often with $k < d$. Similar to a standard autoencoder, the model consists of an **encoder**, which maps the observed data into a latent space, and a **decoder**, which attempts to reconstruct the original observation from its latent representation.

A key difference from a standard autoencoder is that the encoder does not map each observation $x$ to a single deterministic point $z$. Instead, it learns a **distribution** over possible latent representations. A common choice is a Gaussian distribution,
$$
q_\phi(z \mid x)
=
\mathcal{N}\left(
\mu_\phi(x),
\operatorname{diag}(\sigma_\phi^2(x))
\right),
$$
where the encoder neural network, parameterized by $\phi$, predicts a mean $\mu_\phi(x)$ and variance $\sigma_\phi^2(x)$ for each input. We then sample a latent representation $z$ from this distribution.


> **A bit of some tricky math!!** To allow gradients to propagate through this sampling operation, VAEs use the \textbf{reparameterization trick}. Rather than sampling the latent $z$ directly, we sample a much easier normal noise variable $\epsilon \sim \mathcal{N}(0,I)$. Since $z$ is gaussian i.e., $z=\mu_\phi(x)+\sigma_\phi(x) \odot \epsilon$, randomness is therefore isolated in $\epsilon$, while $z$ remains a differentiable function of the encoder outputs.

The decoder, parameterized by $\theta$, then models the distribution that describes how an observation $x$ can be generated from its latent representation $z$:
$$
p_\theta(x \mid z),
$$

During training, the VAE tries to maximize the probability of the data $$p(x)$$, which would essentially equate to : "Maximize the model that fits our observed data $x$". However, this is actually impossible to maximize. So to do this, a VAE instead learns parameters $\phi$ and $\theta$ and maximizes the *evidence lower bound (ELBO)*,
$$
\mathcal{L}_{\mathrm{ELBO}}(x)
=
\mathbb{E}_{q_\phi(z\mid x)}
\left[
\log p_\theta(x\mid z)
\right]
-
D_{\mathrm{KL}}
\left(
q_\phi(z\mid x)
\,\|\, p(z)
\right),
$$

The prior (= assumed distribuiton) for the latent vector $z$ is typically chosen as Gaussian: $p(z) = \mathcal{N}(0,I)$.


The first term in the ELBO ($\mathbb{E}_{q_\phi(z\mid x)}\left[\log p_\theta(x\mid z)\right]$)  encourages the latent representation to preserve information needed to reconstruct $x$. The second term (the KL part, which stands for Kullback–Leibler divergence) encourages the learned latent distributions to remain close to the prior distribution. Together, these objectives encourage the VAE to learn a latent space that preserves important information about the input while remaining structured enough to sample from and generate new observations.

After training, the encoder can therefore be used as a representation-learning model: for each observation $x$, quantities such as the posterior mean $\mu_\phi(x)$ can be used as a learned feature representation for downstream tasks such as prediction, clustering, or visualization.

Also: Note that the above VAE assumed the inputs were 

## <a id="proj3"></a> Project 3: Evaluating Numerical Preprocessing Methods for Tabular Representation Learning 
AI Experience Level: **Intermediate  🧩**

Project Type: **Evaluation**

Last updated: **September 29, 2026** 

One challenge with tabular data is deciding how different types of variables should be represented before they are input into a model. Unlike images, where pixels have a relatively uniform structure, or text, where words or subwords can be represented as discrete tokens in a vocabulary, tabular data is complex and contains many different variable types. While categorical variables (e.g., medications, diagnoses) can often be represented using one-hot vectors or learned embeddings, there is less consensus about the best way to represent numerical features (e.g., weight, heart rate).

In this project, we will specifically study how to best represent numerical features for representation learning models, specfically using a [Variational Autoencoder (VAE)](https://arxiv.org/pdf/1312.6114). VAEs are a classic representation learning method that compresses an input into a lower-dimensional latent representation and then reconstruct the original input using some cool mathematical tricks. There exist several VAEs that are adapted to tabular data (e.g. [TVAE](https://arxiv.org/pdf/1907.00503)), and common approaches for preprocessing numerical data include simple standardization, quantile transformations, discretization or binning, mode-specific normalization using Gaussian mixtures, and newer numerical embedding methods such as piecewise-linear encodings. 

The goal is to understand how this design choice affects what the VAE learns about the data. We will compare the resulting latent representations based on how useful they are for downstream prediction, how much numerical information they preserve, and if different encoding methods have tradeoffs for different input distributions. Ultimately, we want to ask whether there is one broadly effective encoding strategy or whether the best numerical representation depends on the structure of the data.

Project Goals
1. **Understand common approaches for representing numerical variables** in tabular deep learning and specifically VAEs.
2. **Evaluate how numerical encoding affects the quality of patient representations.** For our application to the eICU dataset, we will focus on evaluating patient representations based on if it can be used to properly identify of patient cohorts or predict certain conditions. 
3. **Identify tradeoffs in encoding methods**: do some encodings better preserve numerical information while others produce more useful downstream representations, and do these differences depend on the distributions of the numerical variables? 


### Week 0 (9/21 - 9/27): 
**Onboarding:** 
 - Follow the [instructions provided under "Getting Started"](../README.md) to get setup with the code and data.

**Readings:** 
- [Representation Learning for Tabular Data: A Comprehensive Survey](https://arxiv.org/pdf/2504.16109)
- [Variational Autoencoder (VAE) original paper](https://arxiv.org/pdf/1312.6114) -- I personally recommend supplementing with these [lecture slides](https://sebastianraschka.com/pdf/lecture-notes/stat453ss21/L17_vae__slides.pdf) and this [blog post](https://www.datacamp.com/tutorial/variational-autoencoders). Feel free to use AI to help you understand. You don't need to understand all of the math but you should grasp what the math means and how it's trained.
- [Revisiting Deep Learning Models for Tabular Data](https://arxiv.org/pdf/2106.11959) -- note this work is about prediction tasks and not representation learning, although the challenges with tabular data are still applicable to your project. 

**Assignments:** 
* See course website for what is due. You will typically have a progress report due Saturday evening and often an assignment due Wednesday morning. This week you have a progress report due Saturday. Since it is your first week, you may not have much to report. 

---

### Week 1 (9/28 - 10/3):

**Goals**: 
* Understand the structure of the eICU dataset, including available features, outcomes, and hospital 
* Become familiar with the different distributions that numerical variables can have in real tabular data. (For example, highly skewed, gaussian, contains outliers, or contain many repeated values.)
* Understand how prior works have represented numerical data, and begin thinking about: what information about a numerical variable should a good patient representation preserve? For example, should nearby values remain nearby? Should extreme values remain distinguishable? Should multi-modal structure be preserved?



**Readings**: In some of these papers, the methodology may be challenging. Prioritize understanding the motivation and intuition. You can use AI to help you understand the challenging parts. 
- [Modeling Tabular Data using Conditional GAN](https://arxiv.org/abs/1907.00503) - Focus on Mode-Specific Normalization. Understand why the authors argue that simply scaling a continuous feature may be inadequate sometimes. You don't need to know about GANs, only the VAE part. 
- [On Embeddings for Numerical Features in Tabular Deep Learning](https://arxiv.org/abs/2203.05556) - Focus on the intuition behind piecewise-linear encoding (PLE) and how this differs from giving the model a single standardized scalar.
- [Representation Learning for Tabular Data: A Comprehensive Survey](https://arxiv.org/abs/2504.16109) - Revisit this paper from last week. Revisit the sections discussing the unique challenges of tabular features and feature representations. Focus on how numerical and categorical features differ from other data modalities.
- [Binning as a Pretext Task: Improving Self-Supervised Learning in Tabular Domains](http://proceedings.mlr.press/v235/lee24v.html?utm_source=chatgpt.com) - Focus on the intuition behind quantile binning.


**Assignments & Tasks**: 
- Review lecture slides for this week.
- By Wednesday, assuming you applied for it last week as expected, you should have been granted access by PhysioNet to the full eICU dataset. If you have not received an email by then, then email me and include the date you applied.
- After you get data access, follow the rest of the instructions on ``Getting Started`` to download the data.
- Then you should run the notebook `week1_explore_eicu_data.ipynb`. This achieves two purposes: 
   - First, you should have a good understanding of the underlying dataset, what features are available, how they are reprsented, and what clinical labels exist. This will help you a lot as the quarter progresses. 
   - Second, this notebook will generate the dataframe `../data/clean_dataset.parquet` which you will need for the project! 
- Suggested but optional exploration: Using the prepared eICU dataset, choose approximately 8–10 numerical variables. Analyze and plot their distributions. Are there different patterns for what numerical variables look like?  
- For Assignment 1 (due next week): This should be done solo. All other assignments will be done in your group. 
   - Please reread the paper from this week [On Embeddings for Numerical Features in Tabular Deep Learning](https://arxiv.org/abs/2203.05556)  (for Part A: Read a Paper)
   - Turn in your outputs of section *3. Section Starter: Now it's your turn!* in `week1_explore_eicu_data.ipynb` as a pdf (as this assignment's Part 2: Section Starter Task). 
   - Ideally, you'd have finished this by 10/4 Week 2 Monday so you don't get behind for next week, but no worries if not :) 
* For Progress Report 1 (due Saturday): Meet with your project group and submit what you all want to accomplish for Week 2. [See the website](https://web.stanford.edu/class/cs197/assignments/project.html#progress-reports) for how we expect project reports to be structured.


---

### Week 2 (10/5 - 10/11):


**Goals:**
- Finalize input features $X$ and the fixed train/validation/test splits that will be used throughout the project.
- Understand the different types of numerical distributions that occur in the eICU dataset, including approximately Gaussian, skewed, multi/bimodal, zero-inflated, and highly concentrated variables.
- Implement at least six numerical encoding methods that we will compare throughout the project.
- Develop intuition for the pros/cons of each encoding method for different types of numerical distribution. 

**Readings:**
- Revisit mode-specific normalization in [Modeling Tabular Data using Conditional GAN](https://arxiv.org/abs/1907.00503) 
- Revisit [On Embeddings for Numerical Features in Tabular Deep Learning](https://arxiv.org/abs/2203.05556) and piecewise-linear encoding (PLE). Then look at periodic embedding -- how is it trainable?
- [Representation Learning for Tabular Data: A Comprehensive Survey](https://arxiv.org/abs/2504.16109) - Revisit this paper from last week. Revisit the sections discussing the unique challenges of tabular features and feature representations. Focus on how numerical and categorical features differ from other data modalities.
- Revisit [Binning as a Pretext Task: Improving Self-Supervised Learning in Tabular Domains](http://proceedings.mlr.press/v235/lee24v.html?utm_source=chatgpt.com) and quantile binning.
- Read this [blog post](https://towardsdatascience.com/scaling-numerical-data-explained-a-visual-guide-with-code-examples-for-beginners-11676cdb45cb/?source=rss----7f60cf5620c9---4).

**Assignments & Tasks:**
- I have a helper notebook `week2.ipynb` in your project folder to structure this analysis, which will not be due but meant to guide your experiments. 
   - First, let's study numerical distributions in the eICU dataset. Plot their distributions and identify examples that are approximately Gaussian, skewed, ..etc. 
      - Pick several representative variables that you will continue using as examples throughout the project.
   - Implement several (ie. 6+) encoding methods behind a common interface: standardized scaler, quantile binned, min max binned, mode specific normalization, PLE, periodic encoding, log transformed, quantile normalzed 
   - Apply the encoding methods to the variables and plot the resulting distributions. Which methdos seem to encode each variable type best? i.e. "quantile binned works best for Gaussian variables.." 
   - Now that we've selected our numerical features we care about, pick several categorical features and finalize the full set of features $X$ (X should contain around ~20 features) we will model. Finalize any patient inclusion/exclusion criteria, and fix the train/val/test data splits. 
- Review lecture slides for this week.
- Assignment 1 due Wednesday. Progress Report 2 due Saturday, and should describe your experiments this week. 
- Assignment 2: Related Work assigned. This and all future handins are a group assignment. Read the description on the website. [Nearest neighbor papers listed here.](https://docs.google.com/document/d/10Qe-m0KK5pyykERt7R2zzxDdnpgx3WI7D3OtGlFqDv4/edit?usp=sharing)

---

### Week 3 (10/12 - 10/18): 

**Goals:**
- Understand how handling missing numerical values will inform the downstream distribution. 
- Go over the VAE code / training pipeline.
- Implement and train a simple tabular VAE using standardized numerical inputs. Track its loss functions and understand what "training" looks like. What does reconstruction loss and KL loss refer to? 
- Get the VAE to train well! 
- Start understanding how to probe the learned latent representation: what makes a good representation, and how do you test for that? 

**Readings:** Understand the basics of VAE and how to test the learned representation! 
- Read the [brief intro to VAEs I wrote above](#vae)
- Original [Variational Autoencoder (VAE) paper](https://arxiv.org/pdf/1312.6114) -- can feed into AI to help as it's a lot of math.
- If you are confused: Two helpful blog posts: [Lilian Weng's](https://lilianweng.github.io/posts/2018-08-12-vae/) and this [blog post](https://www.datacamp.com/tutorial/variational-autoencoders). Also these [lecture slides](https://sebastianraschka.com/pdf/lecture-notes/stat453ss21/L17_vae__slides.pdf).

**Assignments & Tasks:**
- I have a helper notebook `week3.ipynb` in your project folder to structure this analysis, which will not be due but meant to guide your experiments. 
   - Understand the provided code for the tabular VAE.
   - Brief detour into handling missing values! This is actually super relevant for your project as this will inform what the numerical distribution looks like after.
   - First, train a basic  tabular VAE. Instead of training all of $X$ which is mixed type, let's start with only the standardized numerical features. 
      - Debug / play with parameters until the training and validation loss go down. Experiment with different hyperparameters. Plot using notebook or using the online site Weights and Biases (see notebook).
   - Second, do some very simple experiments for the quality of a latent variable $z$ to understand the question of "what makes a good representation". 
- Review lecture slides for this week.
- Assignment 2 due Wednesday. Progress Report 3 due Saturday, and should describe your experiments this week. Include all figures from your VAE training runs! 
- Assignment 3: Introduction assigned. Read the description on the website.

---

### Week 4 (10/19 - 10/25): 

**Goals:**


**Assignments & Tasks:** Note, no more helper notebooks.

- In your progress report, include all figures and hypotheses. You should be able to provide an answer to: ...
- Review lecture slides for this week.
- Assignment 3 due Wednesday. Progress Report 4 due Saturday, and should describe your experiments this week. 

---

### Week 5 (10/26 - 11/1): 

**Goals:**

**Assignments & Tasks:** 

* Review lecture slides for this week.
* Progress Report 5 due Saturday.

---

### Week 6 (11/2- 11/8): 

**Goals:**

**Assignments & Tasks:** 

- Review lecture slides for this week.
- Progress Report 6 due Saturday.


---

### Week 7 (11/9- 11/15): 

**Goals:**


**Assignments & Tasks:** 
- Review lecture slides for this week.
- Progress Report 7 due Saturday. Last one!
- Assignment 4: Evaluation Plan is released. Check the webpage for details. [Note: Assignment 5: Draft Paper due in 2 weeks during Thanksgiving break.]



---
### Week 8 (11/16- 11/22): 

**Goals:**

**Assignments / Project Tasks:**

- Produce some final figures and tables needed for the paper -- you will need this for submitting the Draft Paper! 
- Review lecture slides for this week.
* No Progress Report!
* Assignment 4: Evaluation Plan is due Wednesday.
* Assignment 5: Draft Paper is released and due next Wednesday. Check webpage for details. Plan accordingly.


---
### THANKSGIVING BREAK (11/23- 11/29): 
For this week, you can optionally get started on Week 9 so you have more time to write/prepare next week, or you can take the week off but have to do experiments and writing next week. It is up to you. 

**Assignments & Tasks:** 
* You can review the lecture slides either this week or next week.
* No Progress Report!
* Assignment 5: Draft Paper is due Wednesday.
* Assignment 6: Draft Talk is released and due next Wednesday. Check webpage for details. 


---
### Week 9 (11/30- 12/6): 
**Goals:**: 
- Finalize the scientific interpretation of the results.
- Finalize all final figures and tables needed for the paper.
- Turn the project into a coherent research paper & research talk. You got this!

**Assignments & Tasks:** 

* Finalize the main figures and tables for the paper.
* Review lecture slides for Publication and Peer Review.
* No Progress Report!
* Assignment 6: Draft Talk is due Wednesday.

---

### Week 10 (12/7- 12/11): 
**Goals:**: 
- Turn the project into a coherent research paper & research talk. You got this!

**Assignments & Tasks:** 
* Final Talk in class 12/11 during the final exam slot. Final Paper due evening of 12/11. See website for more details on what is expected. 


## <a id="proj4"></a> Project 4: Information Retention in Tabular Representation Learning 

AI Experience Level: **Advanced 🌀**

Project Type: **Evaluation**

This project focuses on the question: what information do representation-learning methods supress and retain with a limited representation capacity? Tabular data is especially interesting with respect to this question because different features can have very different statistical properties: for example, there may be high-variance continuous variables, sparse or highly skewed categorical variables, or categorical features with hundreds of possible values. Some of these features may be clinically useful, while others may contain little or no information about the outcome we ultimately care about.

In this project, we will compare two different representation-learning objectives: a [Variational Autoencoder (VAE)](https://arxiv.org/pdf/1312.6114), which learns a latent representation by reconstructing the original input, and [SCARF](https://arxiv.org/pdf/2106.15147), a contrastive method that learns similar representations for different corrupted versions of the same observation. Using a the eICU dataset, we will append synthetic nuisance features whose statistical properties we can control, for example, high-variance numerical variables or high-cardinality categorical variables.

The goal is to understand if different representation-learning objectives (i.e., reconstructive versus contrastive) exhibit different forms of feature suppression by varying the latent dimension and probing the utility of the latent representation. 

The four broad project goals are: 

1. **Understand reconstructive and contrastive objectives in representation learning**: what do VAE and SCARF capture based on their training objective? 
2. **Capture information suppression / retention in representation learning**: first, how do we measure feature supression and retention?; second, how do VAE and SCARF compare by these metrics? 
3. **Understand tabular input distribution and information retention**: how do different methods respond differently to high-variance, high-cardinality, or highly skewed nuisance features in tabular data?
4. **Linking feature retention to downstream utility:** do nuisance features reduce how much clinically relevant information about a fixed outcome $Y$ survives in the learned representation?


### Week 0 (9/21 - 9/27): 
**Onboarding:** 
 - Follow the [instructions provided under "Getting Started"](../README.md) to get setup with the code and data.

**Readings:** 
- [Representation Learning for Tabular Data: A Comprehensive Survey](https://arxiv.org/pdf/2504.16109)
- [Variational Autoencoder (VAE) original paper](https://arxiv.org/pdf/1312.6114) -- I personally recommend supplementing with these [lecture slides](https://sebastianraschka.com/pdf/lecture-notes/stat453ss21/L17_vae__slides.pdf) and this [blog post](https://www.datacamp.com/tutorial/variational-autoencoders). Feel free to use AI to help you understand. You don't need to understand all of the math but you should grasp what the math means and how it's trained.
- [SCARF: Self-Supervised Contrastive Learning using Random Feature Corruption](https://arxiv.org/pdf/2106.15147)

**Assignments:** 
* See course website for what is due. You will typically have a progress report due Saturday evening and often an assignment due Wednesday morning. This week you have a progress report due Saturday. Since it is your first week, you may not have much to report. 


### Week 1 (9/28 - 10/3):

**Goals**: 
* Understand the structure of the eICU dataset, including available features, outcomes, and hospital identifiers.



**Readings**: In some of these papers, the methodology may be challenging. Prioritize understanding the motivation and intuition. You can use AI to help you understand the challenging parts. 
- [Can Contrastive Learning Avoid Shortcut Solutions?](https://arxiv.org/abs/2106.11230) - Focus on the idea of feature suppression. Understand why improving representation of one feature can sometimes hurt representation of another.
- [Which Features are Learnt by Contrastive Learning? On the Role of Simplicity Bias in Class Collapse and Feature Suppression](https://arxiv.org/abs/2305.16536) - Understand what is simplicity bias, i.e., why might a model learn an easy but unimportant feature before a harder, useful feature? Pay particular attention to the authors' discussion of embedding dimensionality and data augmentation as factors that affect feature suppression.
- [Why do tree-based models still outperform deep learning on tabular data?](https://arxiv.org/abs/2207.08815) - Focus on the parts discussing informative feature learning.  
- [Addressing Feature Suppression in Unsupervised Visual Representations](https://ieeexplore.ieee.org/document/10030871) - Use this paper mainly as an example of how feature suppression can be studied experimentally.



**Assignments & Tasks**: 
- Review lecture slides for this week.
- By Wednesday, assuming you applied for it last week as expected, you should have been granted access by PhysioNet to the full eICU dataset. If you have not received an email by then, then email me and include the date you applied.
- After you get data access, follow the rest of the instructions on ``Getting Started`` to download the data.
- Then you should run the notebook `week1_explore_eicu_data.ipynb`. This achieves two purposes: 
   - First, you should have a good understanding of the underlying dataset, what features are available, how they are reprsented, and what clinical labels exist. This will help you a lot as the quarter progresses. 
   - Second, this notebook will generate the dataframe `../data/clean_dataset.parquet` which you will need for the project! 
- For Assignment 1 (due next week): This should be done solo. All other assignments will be done in your group. 
   - Please reread the paper from Week 0 [Addressing Feature Suppression in Unsupervised Visual Representations](https://ieeexplore.ieee.org/document/10030871) (for Part A: Read a Paper)
   - Turn in your outputs of section *3. Section Starter: Now it's your turn!* in `week1_explore_eicu_data.ipynb` as a pdf (as this assignment's Part 2: Section Starter Task). 
   - Ideally, you'd have finished this by 10/4 Week 2 Monday so you don't get behind for next week, but no worries if not :) 
* For Progress Report 1 (due Saturday): Meet with your project group and submit what you all want to accomplish for Week 2. [See the website](https://web.stanford.edu/class/cs197/assignments/project.html#progress-reports) for how we expect project reports to be structured.

---
### Week 2 (10/5 - 10/11):


**Goals:**


**Readings:**


**Assignments & Tasks:**
- I have a helper notebook `week2.ipynb` in your project folder to structure this analysis, which will not be due but meant to guide your experiments. 
   
- Review lecture slides for this week.
- Assignment 1 due Wednesday. Progress Report 2 due Saturday, and should describe your experiments this week. 
- Assignment 2: Related Work assigned. This and all future handins are a group assignment. Read the description on the website. [Nearest neighbor papers listed here.](https://docs.google.com/document/d/10Qe-m0KK5pyykERt7R2zzxDdnpgx3WI7D3OtGlFqDv4/edit?usp=sharing)

---

### Week 3 (10/12 - 10/18): 

**Goals:**
- Go over the VAE code / training pipeline.
- Implement and train a simple tabular VAE using standardized numerical inputs. Track its loss functions and understand what "training" looks like. What does reconstruction loss and KL loss refer to? 
- Get the VAE to train well! 
- Start understanding how to probe the learned latent representation: what makes a good representation, and how do you test for that? 

**Readings:** Understand the basics of VAE and how to test the learned representation! 
- Read the [brief intro to VAEs I wrote above](#vae)
- Original [Variational Autoencoder (VAE) paper](https://arxiv.org/pdf/1312.6114) -- can feed into AI to help as it's a lot of math.
- If you are confused: Two helpful blog posts: [Lilian Weng's](https://lilianweng.github.io/posts/2018-08-12-vae/) and this [blog post](https://www.datacamp.com/tutorial/variational-autoencoders). Also these [lecture slides](https://sebastianraschka.com/pdf/lecture-notes/stat453ss21/L17_vae__slides.pdf).

**Assignments & Tasks:**
- I have a helper notebook `week3.ipynb` in your project folder to structure this analysis, which will not be due but meant to guide your experiments. 
   - Understand the provided code for the tabular VAE.
   - Brief detour into handling missing values!
   - First, train a basic  tabular VAE. Instead of training all of $X$ which is mixed type, let's start with only the standardized numerical features. 
      - Debug / play with parameters until the training and validation loss go down. Experiment with different hyperparameters. Plot using notebook or using the online site Weights and Biases (see notebook).
   - Second, do some very simple experiments for the quality of a latent variable $z$ to understand the question of "what makes a good representation". 
- Review lecture slides for this week.
- Assignment 2 due Wednesday. Progress Report 3 due Saturday, and should describe your experiments this week. Include all figures from your VAE training runs! 
- Assignment 3: Introduction assigned. Read the description on the website.

---
### Week 4 (10/19 - 10/25): 

**Goals:**


**Assignments & Tasks:** Note, no more helper notebooks.

- In your progress report, include all figures and hypotheses. You should be able to provide an answer to: ...
- Review lecture slides for this week.
- Assignment 3 due Wednesday. Progress Report 4 due Saturday, and should describe your experiments this week. 

---

### Week 5 (10/26 - 11/1): 

**Goals:**

**Assignments & Tasks:** 

* Review lecture slides for this week.
* Progress Report 5 due Saturday.

---

### Week 6 (11/2- 11/8): 

**Goals:**

**Assignments & Tasks:** 

- Review lecture slides for this week.
- Progress Report 6 due Saturday.


---

### Week 7 (11/9- 11/15): 

**Goals:**


**Assignments & Tasks:** 
- Review lecture slides for this week.
- Progress Report 7 due Saturday. Last one!
- Assignment 4: Evaluation Plan is released. Check the webpage for details. [Note: Assignment 5: Draft Paper due in 2 weeks during Thanksgiving break.]



---
### Week 8 (11/16- 11/22): 

**Goals:**

**Assignments / Project Tasks:**

- Produce some final figures and tables needed for the paper -- you will need this for submitting the Draft Paper! 
- Review lecture slides for this week.
* No Progress Report!
* Assignment 4: Evaluation Plan is due Wednesday.
* Assignment 5: Draft Paper is released and due next Wednesday. Check webpage for details. Plan accordingly.


---
### THANKSGIVING BREAK (11/23- 11/29): 
For this week, you can optionally get started on Week 9 so you have more time to write/prepare next week, or you can take the week off but have to do experiments and writing next week. It is up to you. 

**Assignments & Tasks:** 
* You can review the lecture slides either this week or next week.
* No Progress Report!
* Assignment 5: Draft Paper is due Wednesday.
* Assignment 6: Draft Talk is released and due next Wednesday. Check webpage for details. 


---
### Week 9 (11/30- 12/6): 
**Goals:**: 
- Finalize the scientific interpretation of the results.
- Finalize all final figures and tables needed for the paper.
- Turn the project into a coherent research paper & research talk. You got this!

**Assignments & Tasks:** 

* Finalize the main figures and tables for the paper.
* Review lecture slides for Publication and Peer Review.
* No Progress Report!
* Assignment 6: Draft Talk is due Wednesday.

---

### Week 10 (12/7- 12/11): 
**Goals:**: 
- Turn the project into a coherent research paper & research talk. You got this!

**Assignments & Tasks:** 
* Final Talk in class 12/11 during the final exam slot. Final Paper due evening of 12/11. See website for more details on what is expected. 
