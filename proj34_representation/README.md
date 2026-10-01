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
$$q_\phi(z \mid x)=\mathcal{N}\left(\mu_\phi(x),\text{diag}(\sigma_\phi^2(x)\right)),$$
where the encoder neural network, parameterized by $\phi$, predicts a mean $\mu_\phi(x)$ and variance $\sigma_\phi^2(x)$ for each input. We then sample a latent representation $z$ from this distribution.


> **A bit of some tricky math!!** To allow gradients to propagate through this sampling operation, VAEs use the \textbf{reparameterization trick}. Rather than sampling the latent $z$ directly, we sample a much easier normal noise variable $\epsilon \sim \mathcal{N}(0,I)$. Since $z$ is gaussian i.e., $z=\mu_\phi(x)+\sigma_\phi(x) \odot \epsilon$, randomness is therefore isolated in $\epsilon$, while $z$ remains a differentiable function of the encoder outputs.

The decoder, parameterized by $\theta$, then models the distribution that describes how an observation $x$ can be generated from its latent representation $z$:
$$p_\theta(x \mid z),$$

During training, the VAE tries to maximize the probability of the data $p(x)$, which would essentially equate to : "Maximize the model that fits our observed data $x$". However, this is actually impossible to maximize. So to do this, a VAE instead learns parameters $\phi$ and $\theta$ and maximizes the *evidence lower bound (ELBO)*,
$$\mathcal{L}_{\mathrm{ELBO}}(x)=\mathbb{E}_{q_\phi(z\mid x)}\left[\log p_\theta(x\mid z)\right] -D_{\mathrm{KL}}\left(q_\phi(z\mid x)\,\|\, p(z)\right),$$

The prior (= assumed distribuiton) for the latent vector $z$ is typically chosen as Gaussian: $p(z) = \mathcal{N}(0,I)$.


The first term in the ELBO ($\mathbb{E}_{q_\phi(z\mid x)}\left[\log p_\theta(x\mid z)\right]$)  encourages the latent representation to preserve information needed to reconstruct $x$. The second term (the KL part, which stands for Kullback–Leibler divergence) encourages the learned latent distributions to remain close to the prior distribution. Together, these objectives encourage the VAE to learn a latent space that preserves important information about the input while remaining structured enough to sample from and generate new observations.

After training, the encoder can therefore be used as a representation-learning model: for each observation $x$, quantities such as the posterior mean $\mu_\phi(x)$ can be used as a learned feature representation for downstream tasks such as prediction, clustering, or visualization.


## <a id="proj3"></a> Project 3: Evaluating Numerical Preprocessing Methods for Tabular Representation Learning 
AI Experience Level: **Intermediate  🧩**

Project Type: **Evaluation**

Last updated: **September 30, 2026** 

One challenge with tabular data is deciding how different types of variables should be represented before they are input into a model. Unlike images, where pixels have a relatively uniform structure, or text, where words or subwords can be represented as discrete tokens in a vocabulary, tabular data is complex and contains many different variable types. While categorical variables (e.g., medications, diagnoses) can often be represented using one-hot vectors or learned embeddings, there is less consensus about the best way to represent numerical features (e.g., weight, heart rate).

In this project, we will specifically study how to best represent numerical features for representation learning models, specfically using a [Variational Autoencoder (VAE)](https://arxiv.org/pdf/1312.6114). VAEs are a classic representation learning method that compresses an input into a lower-dimensional latent representation and then reconstruct the original input using some cool mathematical tricks. There exist several VAEs designed specifically for tabular data, including [TVAE](https://arxiv.org/pdf/1907.00503), and prior work has proposed many different ways to represent numerical features in deep learning models. In this project, we will compare six strategies:

1. **Standardized scalar**
2. **Quantile-normalized scalar**
3. **Mode-specific normalization using Gaussian mixtures**
4. **Quantile binning**
5. **Piecewise-linear encoding (PLE)**
6. {Your choice}

The goal is to understand how the choice of numerical representation affects the quality and content of the learned latent representation. We will evaluate representations along several dimensions, including: whether they preserve information about the original numerical variables, whether they are useful for downstream clinical prediction, and whether distances in the latent space correspond to meaningful similarities between patients.


To make these comparisons interpretable, we will keep the overall VAE architecture, dataset splits, and training procedure as consistent as possible across experiments and vary primarily the numerical representation. We will also examine whether the relative performance of different encoding strategies depends on the distributional properties of the underlying variables. For example, an encoding that works well for an approximately Gaussian variable may behave differently for a highly skewed or multimodal variable.

Main project question:
> **How does numerical feature representation affect what information a VAE preserves in its latent representation, and do different encoding strategies work better for different types of numerical variables?**

Project Goals
1. **Understand common approaches for representing numerical variables** in tabular deep learning and specifically VAEs.
2. **Evaluate how numerical encoding affects the quality of learned patient representations.** Using the eICU dataset, we will evaluate representations based on downstream prediction, recovery of original numerical information, and latent-space similarity.
3. **Identify tradeoffs between encoding strategies.** Do some encodings better preserve numerical information while others produce more useful downstream representations?
4. **Study whether encoding performance depends on the structure of the numerical variable.** In particular, we will examine whether different methods behave differently for approximately Gaussian, skewed, multimodal, heavy-tailed, or otherwise unusual numerical distributions.


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
   - First, train a basic tabular VAE. Let's start with only the standardized numerical features instead of the the other fancy encoders.  
      - Debug / play with parameters until the training and validation loss go down. Experiment with different hyperparameters. Plot using notebook or using the online site Weights and Biases (see notebook).
   - Second, do some very simple experiments for the quality of a latent variable $z$ to understand the question of "what makes a good representation". 
- Review lecture slides for this week.
- Assignment 2 due Wednesday. Progress Report 3 due Saturday, and should describe your experiments this week. Include all figures from your VAE training runs! 
- Assignment 3: Introduction assigned. Read the description on the website.

---

### Week 4 (10/19 - 10/25): 

**Goals:**
- Understand how to define what a "good" latent representation means, and all the ways it can be useful: for prediction, information preservation, and meaningful patient similarity.
- Evaluate which original numerical variables can be recovered from $z$.
- Experiment and see if distances in latent space correspond to meaningful similarities between patients.

**Readings**: Optional readings that explore how to eval for a good latent representation:
 - **Deep Patient: An Unsupervised Representation to Predict the Future of Patients from the Electronic Health Records** — uses downstream prediction.
 - **Autoencoder-Based Representation Learning for Similar Patients Retrieval From Electronic Health Records: Comparative Study** - looks at distance of patients

**Assignments & Tasks:** Note, no more helper notebooks.
- Use the standardized-input VAE trained last week (you can save the weights locally or on wandb if you'd prefer) and use the encoder mean (mu(x)) to generate the latent z. Generate and save Ztrain, Ztest, Zval.
- **PRED Y**: Train a logistic or linear regression model using Ztrain to predict some downstream outcome (Y) that isn't in X. How does it do? Plot the curves. 
   - Then evaluate this model on Ztest using an appropriate metric such as AUROC, (R^2), or MAE. 
   - You should train the same type of prediction model directly on the original standardized numerical features (X) as a reference. How do they compare??
- **X RECOVERY:** For each numerical feature X_j, train a simple linear regression to see if Ztrain can recover the raw X_j. Plot the results. Which variable does the best with the lowest reconstruction? 
   - Compare these results with the numerical distributions you studied earlier. Are certain types of variables easier or harder to preserve?
- **GEOMETRY:** For a sample of test patients, compute pairwise distances in the original standardized feature space Xtest, and compare that to the distance in latent space. (Ask yourself: why are we doing this?)
   - You can use whatever distane function d() you'd like, i.e. MAE, MSE. You can also decide if you want to look at Xtest all features or only Xtest numerical features. 
   - Ideally the distance in observation space X is similar in latent space Z (i.e. different patients still look different; similar patients still look similar). For patients i,j, compare distance_Zspace(i,j) with distance_Xspace(i,j) for all i,j pairs. Is there a correation? Plot a scatter plot. 
   - Pick a few test patients and inspect their nearest neighbors in (z). Are they similar in age, outcome, or selected clinical measurements?
- **(Optional) PCA**: Make one simple 2D PCA visualization of Z, colored by either (Y), age, or one clinical variable.
- Provide at least one plot comparing across either: latent dim of Z, a different numerical encoding (this will be done next week!), or another hparam you tried. 
- In your progress report, include all figures and hypotheses. You should be able to provide an answer to: what are ways we can evaluate a good latent dimensions? For baseline standardized scaling, how does the latent representation look? Is it good or bad? 
- Review lecture slides for this week.
- Assignment 3 due Wednesday. Progress Report 4 due Saturday, and should describe your experiments this week. 

---

### Week 5 (10/26 - 11/1): 

**Goals:**
- Form clear hypotheses about how the six numerical encoding strategies may affect learned VAE representations.
- Finalize the main experiments, baselines, and evaluation metrics before looking at the results.
- Build a reproducible experimental pipeline that can train and evaluate all six encoding strategies.

**Assignments & Tasks:** 
- Discuss as a group, and then write down your main hypotheses before running the experiments connecting numerical variable types and encoding types to metrics for which we evaluate representations. 
   - For example: Which encodings might work particularly well for skewed or multimodal variables? 
   - Write approximately 3–5 concrete hypotheses that can be tested with your experiments.
- Finalize the six numerical encoding conditions from Week 2. Debug, change, and visualize as you see fit. You may have gained new insight!
- Decide on and freeze the main VAE configuration that will be used across encodings, including latent dimension, hidden layers, optimizer, learning rate, (\beta), epochs/early stopping, and random seeds. These should be parameters that you think will lead to good model training overall
   - **One super important caveat:** When comparing design choices you always want the # learnable parameters to be equal so you aren't confounded by how big the model is. However, some of our encoding methods produce bigger input dimensions than others (consider that StandardScaler would produce K dimensions from K variables, while mode specific normalization or binning increases the number of dimensions) so it wouldn't be a fair comparison. Discuss with your group how to handle this. Can you increase the latent dimension z for the smaller input spaces? 
- Discuss and finalize what quality metrics you will use to evaluate different types of "goodness" for latent representations. Maybe some strategies will be better at some metrics and worse at others. 
   - (i.e. prediction performance feature recovery, and latent space similarity). 
   - Can you make a script such that given a model, this evaluation is easy? 
- Define your baseline comparisons. 
   - Include performance using the original standardized X where appropriate. Are there any others? 
- Create one experiment script or configuration system that can run the same VAE and evaluation pipeline while changing only the numerical encoding method. i.e `python train.py ARGUMENTS`
- Make an experiment table before running anything, listing each model, seed, encoding, and evaluation that needs to be completed.
* Review lecture slides for this week.
* Progress Report 5 due Saturday.

---

### Week 6 (11/2- 11/8): 

**Goals:**
- Run the complete set of main experiments across all six numerical encoding strategies.
- Verify that each VAE trains successfully and that differences are not caused by obvious training failures.
- Evaluate each learned representation using the same metrics developed in Week 4.
- Begin identifying consistent patterns. 

**Assignments & Tasks:** 
- Train a VAE for each of the six numerical encoding strategies using the fixed setup from Week 5.
   - Ideally, you'd also run a few random seeds for each condition.
- Track basic training diagnostics for every run, including reconstruction loss, KL loss, validation loss, and convergence behavior. Plot those side by side. 
- Run the downstream evaluation pipeline for each encoding strategy (i.e. prediction performance feature recovery, and latent space similarity). 
- Inspect results as experiments finish. Identify failed runs, unexpected outliers, or suspicious differences.
- Produce a plot comparing all encodings for each evaluation metric. Summarize the main experiment results across encodings and seeds using tables and confidence intervals/error bars.
- Review lecture slides for this week.
- Progress Report 6 due Saturday.


---

### Week 7 (11/9- 11/15): 

**Goals:**
- Start transitioning from asking "which method performed better?" to "why might we be seeing these differences?" From here you can design experiments to test these!!
- Determine which patterns in the main results are consistent and scientifically interesting.
- Identify possible explanations for why particular numerical encodings perform differently.
- Design a small number of targeted experiments that distinguish between those explanations.

**Assignments & Tasks:** 
- Compare the results with your Week 5 hypotheses. Which were supported? Are there any new hypotheses? 
- Observe: Are there any tradeoffs? For example, does better prediction of Y translate to worse performance on other evaluaiton criteria? Why do you think this is? 
- Examine feature-level scalar recovery. Identify variables for which encoding choice matters substantially in reconstruction. Are these variables mappable to any distribution types from Week 2? 
- Create preliminary versions of the most important figures, including a comparison of representation fidelity and downstream utility across encodings.
- Select approximately 2–3 findings that you want to understand better. How can you test these experimentally?? Design those ablation experiments. 
   - Try to avoid running experiments simply because they are easy to run.
   - Write a short explanation and hypothesis of each proposed ablation. What outcomes would support or contradict the hypothesis?
- Review lecture slides for this week.
- Progress Report 7 due Saturday. Last one!
- Assignment 4: Evaluation Plan is released. Check the webpage for details. [Note: Assignment 5: Draft Paper due in 2 weeks during Thanksgiving break.]



---
### Week 8 (11/16- 11/22): 

**Goals:**
- Run targeted experiments that help explain the main results, as developed last week. 
- Narrow the project to a small number of defensible main findings.

**Assignments & Tasks:**
- Run the targeted ablations selected in Week 7. You should at least 2-3 types of ablations. Keep each ablation simple and change one factor at a time whenever possible.
   - Make sure you run on random seeds and report variation rather than the best run. 
- Investigate whether results differ by numerical-variable type. For example, compare approximately Gaussian, skewed, multimodal, or heavy-tailed variables.
- Choose the main findings that you believe should become the main results of the paper.
- Record important negative results and limitations. Unexpected or null findings are useful if they help clarify when numerical encoding does or does not matter.
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
- Translate experimental results into a clear scientific argument rather than a collection of benchmarks. How can we convey clear conclusions? 
- Finalize all final figures and tables needed for the paper.
- Turn the project into a coherent research paper & research talk. You got this!

**Assignments & Tasks:** 

* Finalize the main figures and tables for the paper. Aim for approximately 3–5 figures/tables that communicate the major findings without showing every experiment.
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

Last updated: **September 30, 2026**

This project focuses on the question: 
> **How do different representation-learning objectives determine what information is retained or suppressed in noisy tabular data, particularly when representation capacity is limited?**

Tabular data is especially interesting because different features can have very different statistical properties: for example, there may be high-variance continuous variables, sparse or highly skewed categorical variables, or categorical features with hundreds of possible values. Some of these features may be clinically useful, while others may contain little or no information about the outcome we ultimately care about.

In this project, we will compare two different representation-learning objectives: a [Variational Autoencoder (VAE)](https://arxiv.org/pdf/1312.6114), which learns a latent representation by reconstructing the original input, and [SCARF](https://arxiv.org/pdf/2106.15147), a contrastive method that learns similar representations for different corrupted versions of the same observation. 

We want to test how **different types of noisy tabular data** affect the quality of a learned representation $Z$. However, a key principle of experimental design is to vary only one factor at a time. If we change multiple things simultaneously, it becomes difficult to determine what actually caused an observed difference. For example, if we add a high-variance feature $X_j$ and observe that the quality of a VAE representation changes, we want to know whether this is because of the **high variance of $X_j$**, rather than because $X_j$ happened to contain important clinical information.

A synthetic setup allows us to control these properties directly. Using the eICU dataset, we will first define a small set of clinically meaningful **signal features** $X$ and construct a synthetic clinical outcome that depends on these features:

$$Y_{\text{synth}} = f(X) + \epsilon.$$

Because we know exactly which features determine $Y_{\text{synth}}$, prediction of $Y_{\text{synth}}$ from the learned representation $Z$ provides one measure of whether clinically relevant signal has been retained.

We will then append synthetic **noisy tabular features** $N$ that contain no additional information about $Y_{\text{synth}}$, but whose statistical properties we can control. We can then vary properties such as high numerical variance, high categorical cardinality, strong skew, or correlation among noisy tabular features while keeping the underlying signal $X$ and outcome $Y_{\text{synth}}$ fixed.

The resulting setup is:

$$X_{input} = [X,N], \qquad Y_{\text{synth}} = f(X) + \epsilon.$$

**where $X_{input}$ is what is input into the VAE / SCARF**. This controlled experiment allows us to make constant the source of information for predicting $Y_{synth}$. By varying different levels of noisy tabular variables (i.e. high cardinality), we can therefore test several things: if certain types of tabular features compete with useful signal for limited representation capacity, whether information about $X$ becomes suppressed in $Z$, and whether this behavior differs between reconstructive and contrastive representation-learning objectives. We can also try varying the latent dimension and probing what information survives in the learned representation $Z$ under stricter capacity.

The four broad project goals are:

1. **Understand reconstructive and contrastive objectives in representation learning:** what do VAE and SCARF capture based on their training objectives?

2. **Capture information suppression / retention in representation learning:** how do we measure whether information about a feature survives in $Z$, and how do VAE and SCARF compare by these metrics?

3. **Understand tabular input distribution and information retention:** how do different methods respond to high-variance, high-cardinality, or highly skewed noisy tabular features?

4. **Link feature retention to downstream utility:** do noisy tabular features reduce how much information about the true signal features $X$, and therefore the synthetic outcome $Y_{\text{synth}}$, survives in the learned representation?


### Week 0 (9/21 - 9/27): 
**Onboarding:** 
 - Follow the [instructions provided under "Getting Started"](../README.md) to get setup with the code and data.

**Readings:** 
- [Representation Learning for Tabular Data: A Comprehensive Survey](https://arxiv.org/pdf/2504.16109)
- [Variational Autoencoder (VAE) original paper](https://arxiv.org/pdf/1312.6114) -- I personally recommend supplementing with these [lecture slides](https://sebastianraschka.com/pdf/lecture-notes/stat453ss21/L17_vae__slides.pdf) and this [blog post](https://www.datacamp.com/tutorial/variational-autoencoders). Feel free to use AI to help you understand. You don't need to understand all of the math but you should grasp what the math means and how it's trained.
- [SCARF: Self-Supervised Contrastive Learning using Random Feature Corruption](https://arxiv.org/pdf/2106.15147)

**Assignments:** 
* See course website for what is due. You will typically have a progress report due Saturday evening and often an assignment due Wednesday morning. This week you have a progress report due Saturday. Since it is your first week, you may not have much to report. 

---
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
### Week 2 (10/4 - 10/11):

**Goals:**
- Understand representation learning and how a latent bottleneck can force a model to retain some information while discarding other information.
- Understand how preprocessing, especially missing-value handling, changes the distribution the VAE learns.
- Understand the basic VAE pipeline: encoder, latent representation Z, decoder, reconstruction loss, and KL loss.
- Train a simple tabular VAE and understand what successful training looks like.
- Introduce the idea of probing a representation: what information is retained in Z and how can we test for it?

**Readings:** Understand the basics of VAE and how to test the learned representation! 
- Read the [brief intro to VAEs I wrote above](#vae)
- Original [Variational Autoencoder (VAE) paper](https://arxiv.org/pdf/1312.6114) -- can feed into AI to help as it's a lot of math.
- If you are confused: Two helpful blog posts: [Lilian Weng's](https://lilianweng.github.io/posts/2018-08-12-vae/) and this [blog post](https://www.datacamp.com/tutorial/variational-autoencoders). Also these [lecture slides](https://sebastianraschka.com/pdf/lecture-notes/stat453ss21/L17_vae__slides.pdf).

**Assignments & Tasks:**
- I have a helper notebook `week2.ipynb` in your project folder to structure this analysis, which will not be due but meant to guide your experiments. 
   - **Understand the provided tabular VAE code.** Find where reconstruction loss and KL loss are calculated, and understand what each encourages the model to do.

   - **Detour into handling missing numerical values.** Examine missingness across the numerical features and decide how missing values will be represented before training.
      - Think about why this choice can affect the learned representation. 
   - **Train a basic tabular VAE.** Categorical values are 1hot encoded, continuous are standard normalized. 
      - Train the VAE and inspect the training and validation curves.
      - Understand what each loss means and what reasonable training behavior should look like.
      - Change a few hyperparameters, such as learning rate, hidden-layer size, or latent dimension. See how these changes affect reconstruction and KL loss.

   - **Run a simple representation probe.** Train a simple linear/logistic model using Z to predict a downstream outcome Y.
       - Compare: Z->Y model vs X->Y model. 
      - Ask: **how much information useful for predicting Y survives after compressing X into Z?**
   - **Optional but recommended**: Make the genreation of Ztrain, Ztest, Zval easy to do and save, and then predict some Y. You will use this in later weeks!
- Review lecture slides for this week.
- Assignment 1 due Wednesday. Progress Report 2 due Saturday, and should describe your experiments this week. 
- Assignment 2: Related Work assigned. This and all future handins are a group assignment. Read the description on the website. [Nearest neighbor papers listed here.](https://docs.google.com/document/d/10Qe-m0KK5pyykERt7R2zzxDdnpgx3WI7D3OtGlFqDv4/edit?usp=sharing)

---

### Week 3 (10/12 - 10/18): 

**Goals:**
- Understand the basic intuition behind contrastive learning and how it differs from reconstruction-based learning.
- Understand how SCARF creates corrupted views of tabular observations and learns representations that are similar across views of the same sample.
- Implement and train SCARF on the same input $X$ used for the VAE.
- Make the VAE and SCARF training pipelines as comparable as possible so later experiments isolate differences in learning objective.
- Compare what the VAE and SCARF representations retain using the same simple downstream Y probe.

**Readings:**  Focus on the intuition behind contrastive learning and SCARF.

- Revisit the original [SCARF paper](https://arxiv.org/pdf/2106.15147). How is a "corrupted X constructed? What is the role of negative examples from other patients? Consider what the contrastive loss is encouraging the representation to retain.
- Revisit [Can Contrastive Learning Avoid Shortcut Solutions?](https://arxiv.org/abs/2106.11230) 
- Revisit [Which Features are Learnt by Contrastive Learning? On the Role of Simplicity Bias in Class Collapse and Feature Suppression](https://arxiv.org/abs/2305.16536) 

**Assignments & Tasks:**
- I have provided a helper notebook, `week3.ipynb`, in your project folder. This notebook is not due; it is meant to guide your experiments.
   - **Understand the provided SCARF code.** Read `scarf_model.py`. Identify the encoder, projection head, corruption procedure, and contrastive loss. Understand what constitutes a positive pair and a negative pair.
   - **Visualize SCARF's corruption of samples X.** Take several example rows of $X$ and generate their corrupted versions X'.
      - Identify which features changed and where the replacement values came from (what distribution?)
      - Experiment with a few corruption rates and understand how they change the difficulty of the learning problem.
   - **Train SCARF on the same X used for the VAE.** Use the same train/validation/test split and data preprocessing as for the VAE.
      - Track training and validation contrastive loss.
      - Experiment with a small number of hyperparameters until training behaves sensibly.
   - **Run the same simple downstream probe for both models.** Like last week for the VAE, train the same simple linear/logistic model to predict Y (or several Ys) from each representation of SCARF, and also from VAE. How do the two methods behave? Is one better than the other? Why do we think this is? 
   - (Optional but recommended to develop over the next two weeks) **Make the VAE and SCARF pipelines easy to run.** You will eventually be retraining the models in week 6 so it should be easy to rerun :) 
      - i.e by calling `python train.py --model VAE --zdim 10` and  by calling `python train.py --model SCARF --zdim 10` or something like that.  NOTE: doing this is NOT in the notebook, you should do this outside of it so you have a nice friendly script to run.
- Review lecture slides for this week.
- Assignment 2 due Wednesday. Progress Report 3 due Saturday, and should describe your experiments this week. Include all figures from your SCARF training runs! 
- Assignment 3: Introduction assigned. Read the description on the website.

---
### Week 4 (10/19 - 10/25): 

**Goals:**
- Define what we mean by feature retention and supression in a learned latent representation $Z$
- Learn how to use simple probes to measure what information can be recovered from $Z$. 
- Compare which features are retained by VAE and SCARF representations , and how to measure retention. 
- Establish a consistent evaluation pipeline that we can reuse later when synthetic noisy tabular features are introduced.

**Readings:**
- Revisit [Can Contrastive Learning Avoid Shortcut Solutions?](https://arxiv.org/abs/2106.11230). Focus on how the authors probe representations for different features and identify tradeoffs between which features are learned.
- Revisit [Which Features are Learnt by Contrastive Learning?](https://arxiv.org/abs/2305.16536). Focus on how the paper defines and measures feature suppression.
- Optional: [Addressing Feature Suppression in Unsupervised Visual Representations](https://openaccess.thecvf.com/content/WACV2023/html/Li_Addressing_Feature_Suppression_in_Unsupervised_Visual_Representations_WACV_2023_paper.html). Look mainly at how controlled experiments are used to determine what information survives in a representation.

**Assignments & Tasks:** Note, no more helper notebooks.
- Start any trained VAE and SCARF encoders you have. You will be retraining in 2 weeks so it doesn't have to be the final models. **Freeze the encoders** so that they are not updated during probing. You can save the weights locally or on wandb. 
- As in prior weeks, generate latent representations for the training, validation, and test sets: Ztrain, Ztest, Zval. Save these for each model. 
- Define **feature retention / supression**. I have provided one def but you all can / should modify or add to it based on what you see have learned:
  > A feature $X_j$ is retained in $Z$ if a simple model can recover information about that feature from $Z$ with MSE <$\tau$. (likely need diff criteria for binary and continuous features)
- For the above def, here would be the eval process: 
   - For each feature $X_j$ train a simple linear model (linear or logistic depending on $X_j$) to predict $Ztrain \rightarrow X_j$. Tune on validation. Report final performance on the held-out test set.
   - Choose an appropriate metric for each feature type: R2 or MSE for continuous; cross entropy or AUPRC for binary
   - Compare every probe against a simple baseline, such as predicting the training-set mean or use a random encoder projection. **This baseline step is actually very important**. 
   - Create a table with one row per feature showing its probe performance under VAE latents and SCARF latents. 
   - Create one main visualization showing feature retention across the two representations. 
- Analyze the resuts for patterns across highly retained / supressed features, and across model types. For ex:
   - what features are strongly retained by both models? By only VAE or SCARF? 
- Use the Y-probe experiment where Z from SCARF and Z from VAE will predict some outcome Y, like mortality. Compare feature-level retention with the Y-probe, and ask if strong prediction of $Y$ requires every original feature to be retained. 
- Write down 2--3 preliminary observations or hypotheses about why certain features appear easier to retain than others. Consider properties such as variance, correlation with other variables, prevalence, or distribution shape.
- **MAIN GOAL:** Save this probing pipeline as reusable code. In later weeks, you should be able to supply a new representation $Z$ and automatically produce the same feature-retention evaluation.
- Review lecture slides for this week.
- Assignment 3 due Wednesday. Progress Report 4 due Saturday, and should describe your experiments this week. Include all figures and hypotheses. 

---

### Week 5 (10/26 - 11/1): 
**Goals:**
- Define and generate signal features X, synthetic outcome Y, and noisy tabular features $N$ for our main experiments!!
- Construct controlled noisy tabular conditions that differ in statistical properties but remain unrelated to (Y).
- Prepare the dataset for the full experimental sweep in later weeks.
- Finalize the main hypotheses about when VAE and SCARF will retain or suppress different features.

**Assignments & Tasks:**
- Select a ~20-dimensional subset of X from prior weeks (you can use the original size but I worry it will take a long time to train models). This is your signal features X!
- We want to create a synthetic outcome that we can then test our Z on, where we are sure that noisy tabular synthetic $N$ is independent of the outcome. Create a synthetic outcome for some linear function $f$: $Y_{synth} = f(X) + \epsilon$. Or perhaps a binary where you take the softmax of the above: $Y_{synth} = \text{softmax}(f(X) + \epsilon)$. Choose parameters so that $Y$ has reasonable prevalence and is neither trivial nor impossible to predict. (i.e. if Y is binary, then its mean should be between 0.1-0.9.)
   - Verify that a supervised model trained directly on $X$ can predict $Y_{\text{synth}}$ reasonably well. [So then a VAE / SCARF on just $X$ should also be able to predict $Y_{\text{synth}}$!]
- Construct synthetic noisy tabular variables  $N$  that are generated independently. I provided code for this in at the end of `scarf_model.py`. In your experiments, you will allow  $N$  to vary by tabular data complexity, such as:
  - **Control:** no additional noisy tabular variables.
  - **Gaussian**: Nothing special, just a normally distributed variable. 
  - **High-cardinality categorical:** independent categorical features with many possible values.
  - **Highly skewed categorical:** independent categorical features where a small number of categories dominate.
  - Any other ideas? 
- Visualize each variable type  $N$  might be. 
- Ask yourself: After we preprocess the data, what does it look like? How does this affect the input dimension into the VAE / SCARF? How does this affect the output dimension? (hint: the answer is different for SCARF and VAE!)
- Create 4 types of $X_{input}=$ with the real eICU dataset $X$, for some number of M extra variables of your choice (I recommend around M=3 new variables):
   - $X_{input}=$ X (control); 
   - $X_{input}=$ X plus M Gaussian variables N; 
   - $X_{input}=$ X plus M high cardinality cateogrical variables N; 
   - $X_{input}=$ X plus M high skewed variables N.
   - any other ideas?
- Save each dataset condition using the **same patients, X, same Y, and same train/validation/test split** so that later comparisons differ only in the noisy tabular features.
- Write 2--4 concrete hypotheses **before running the full experiment**. For example:
  - Smaller latent dim of Z will increase competition between signal and noisy tabular information.
  - VAE and SCARF will differ in which noisy tabular properties they preferentially retain.
  - Increased noisy tabular retention will be associated with reduced signal retention and worse linear probe on $Y_{\text{synth}}$ performance, but only for highly skewed variables. 
* Review lecture slides for this week.
* Progress Report 5 due Saturday.

---
### Week 6 (11/2- 11/8):

**Goals:**
- Finalize all experimental design, including baselines: latent-dimension sweep, noisy tabular conditions, and experimental controls.
- Make sure the complete experimental pipeline can run reliably before launching it at scale for the final experiments.

**Assignments & Tasks:**
- Decide on all VAE / SCARF hyperparameter choices for your experiments. How should the VAE and SCARF be trained such that they are roughly trained similarly?
   - Decide and fix the main preprocessing mechanisms, encoder architecture details, optimizer details, etc. 
- Decide what hyperparameters we want to test, for example: 
   -  Consider different random seeds, including resampling the noisy tabular parameters N. 
   - Try varying the latent dimensions e.g. $dim_Z\in\{4,8,16,28\}$ where $dim_Z$ should always be less than the preprocessed input.
- Finalize what evaluation metrics we will care about and how we will compare them.
   -  *Linear probe of $Y_{\text{synth}}$*: train the same type of supervised model to predict $Y_{\text{synth}}$. Decide on what your evaluation metric will be (R2? MSE? AUROC?). For input, we have all latent results produced by our four variations of $X_{input}$.
   - *Feature retention / supression* of all $X_j$ in the true eICU signal features $X$, as we defined in Week 4. 
   - NOTE: These are pretty much testing the same thing, ie. if a model can predict some feature from Z. 
   - Make sure you can run these eval metrics easily.
- Test briefly to make sure you can run complete VAE and SCARF experiment end-to-end. Make sure training, representation extraction, probing, and result saving all work automatically.
- Run the simplest version of the linear probe baseline of $Y_{\text{synth}}$: train four models, where for each of the four above variations of $X_{input}$ you train the same supervised model directly on the raw features X + $N$ to predict $Y_{\text{synth}}$. 
   - Confirm that performance remains approximately constant across noisy tabular conditions. If it changes substantially, investigate why.
- Create a table listing every experiment to be run in Week 7, including model, variations of $X_{input}$, $dim_Z$, and seed.
- Finalize the baselines that will be used throughout the experiment. (Baselines are important to establish bc they indicate success -- if our method does better than them, or sometimes if it is comparable to an oracle baseline). Some ideas:
  - For the linear probe task, compare to a **X-input supervised baseline:** $X\rightarrow Y$
  - For latent variables Z to test we have 2 baselines: 
      - **No-noisy tabular baseline:** train VAE and SCARF on clean $X$ to measure retention before adding $N$ .
      - **PCA baseline:** compress $X$ to the same $dim_Z$ values and run the same $Z\rightarrow X_j$, and $Z\rightarrow Y_{\text{synth}}$ probes.
- Review lecture slides for this week.
- Progress Report 6 due Saturday.
---

### Week 7 (11/9- 11/15): 


**Goals:**
- Run all experiments!! ie. the complete VAE × SCARF × variations of $X_{input}$ × dimZ experiment. 
- Collect representation-retention and downstream-utility metrics consistently.
- Check that observed patterns are reproducible across random seeds.

**Assignments & Tasks:**
- Train VAE and SCARF for every experiment listed in the Week 6 experimental table.
   - Repeat each experiment across the chosen random seeds. Monitor training curves. 
   - Save model checkpoints, learned representations $Z$, training curves, hyperparameters, and seed for every run.
   - If you are finding these experiments take a long time to run, lmk (i.e. if it would take over a week to train all variations)
- Run the same evaluation metrics (Y probe + feature retention ) from Week 4 on every learned representation.
- Build one master results table containing the model, variations of $X_{input}$, dimZ, seed, probe metrics, downstream performance, and training metrics.
- Summarize results across seeds using the mean and variability rather than focusing on individual runs.
- Check for failed runs, extreme outliers, or suspicious conditions and rerun only when there is a clear technical reason.
- Review lecture slides for this week.
- Progress Report 7 due Saturday. Last one!
- Assignment 4: Evaluation Plan is released. Check the webpage for details. [Note: Assignment 5: Draft Paper due in 2 weeks during Thanksgiving break.]



---
### Week 8 (11/16- 11/22): 


**Goals:**
- Identify patterns in the main experiment -- across variations of $X_{input}$, across $dim_Z$, and across latent model types. How do these source of variations affect the latent evaluation quality? 
- Connect experimental results to feature supression. 
- Use a small number of targeted follow-up experiments to investigate **why** the observed patterns occur.

**Assignments & Tasks:**
- Analyzing experimental results should be framed by several guiding questions or motivations.
- First, **how do different noisy tabular variables "induce" feature suppression?** Across each type latent $Z$ built from variations of $X_{input}$, 
   - Compare the retention performance of $X$.
   - Compare performance predicting $Y_{\text{synth}}$
  - Ask whether some noisy tabular distributions lead to systematically greater suppression than others.

- **How does the representation-learning objective affect feature suppression?**
  - Compare VAE and SCARF under input of **same variation of $X_{input}$** (so now holding fixed $X_{input}$) and the same latent dimension.
  - Compare both feature-retention metrics and $Z \rightarrow Y_{\text{synth}}$ performance.
  - Identify where the two objectives behave differently and develop hypotheses for why.

- Examine the relationship between **signal retention and downstream utility**. When $Z \rightarrow X_j$ decreases, does $Z \rightarrow Y_{\text{synth}}$ also decrease? Is this what you would expect. 

- Examine whether feature competition becomes stronger as $dim_Z$ decreases. Does increasing representation capacity reduce or eliminate suppression?

- Examine feature-level results. Are particular clinical signal features (i.e. lab values, binary variables, etc) consistently suppressed over others?

- Identify the **one or two strongest or most surprising findings** that require further explanation.

- Choose one targeted ablation motivated by those findings. Design a targeted follow-up experiments that test a specific explanation for the observed result. For example, you may have a hypothesis in which you'll need to vary:
  - noisy tabular-block size (i.e. how many variables in $N$);
  - degree of zero inflation, skew, or cardinality;
  - SCARF corruption rate;

- Design at least one **matched comparison** where VAE and SCARF models, and/or different choices of $N$, are matched as closely as possible in input dimensionality and model capacity. This helps ensure observed differences are due to the noisy tabular structure or learning objective rather than a simple difference in dimensionality or parameter count.

- Begin drafting the main result figures. For each figure, write 1–2 sentences stating **what the figure shows** and **what conclusion it supports** -- you will need this for submitting the Draft Paper! 
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
- Translate experimental results into a clear scientific argument rather than a collection of benchmarks. How can we convey clear conclusions? 
- Finalize all final figures and tables needed for the paper.
- Turn the project into a coherent research paper & research talk. You got this!

**Assignments & Tasks:** 
* Finalize the main figures and tables for the paper. Aim for approximately 3–5 figures/tables that communicate the major findings without showing every experiment.
- Draft the Discussion:  When representation capacity is limited, how do reconstructive and contrastive objectives differ in what information they retain or suppress, and how does this affect downstream utility?
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
