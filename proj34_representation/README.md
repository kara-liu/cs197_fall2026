# Project 3 & 4: Representation Learning

**PLEASE DO NOT SHARE THIS MATERIAL BEYOND CS197 AT STANFORD UNIVERSITY**

Machine learning (ML) has achieved major successes on text and images, supported by architectures designed for those data types. Tabular data remains both extremely common and comparatively challenging: electronic health records, surveys, census data, financial records, and many scientific datasets are naturally represented as tables. These datasets often mix continuous, binary, and categorical variables, contain missing values, and include features with very different scales, distributions, frequencies, and relationships with one another.

Separately, the field of **representation learning** asks if we can transform a complex, high-dimensional observation (e.g., an image, large text response, or tabular datapoint) into a smaller latent representation $^{[0]}$ that still captures useful information. Unlike a model trained only to predict one outcome, a useful representation can potentially support many downstream tasks.

In healthcare, for example, we might want a compact patient representation that summarizes information from demographics, laboratory measurements, diagnoses, medications, and vital signs. The same representation could then be used for tasks such as disease risk prediction or identifying similar patients.

The projects are as follows, with the hyperlinks attached:

- [Project 3 (Intermediate Level  🧩): Evaluating Numerical Preprocessing Methods for Tabular Representation Learning](#proj3)
- [Project 4 (Advanced Level 🌀): Information Retention in Tabular Representation Learning ](#proj4)

$^{[0]}$ The terms *latent*, *embedding*, and *representation* are used often interchangeably to denote a multivariate, often compressed, vector $z = f(x)$ of an input $x$, for some function $f$. Some works may also use the term *embedding* to refer to the output of preprocessing an input, i.e., a numerical embedding will be input into a model. 


## A Brief Intro Into VAEs

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
\]

The first term in the ELBO ($\mathbb{E}_{q_\phi(z\mid x)}\left[\log p_\theta(x\mid z)\right]$)  encourages the latent representation to preserve information needed to reconstruct $x$. The second term (the KL part, which stands for Kullback–Leibler divergence) encourages the learned latent distributions to remain close to the prior distribution. Together, these objectives encourage the VAE to learn a latent space that preserves important information about the input while remaining structured enough to sample from and generate new observations.

After training, the encoder can therefore be used as a representation-learning model: for each observation $x$, quantities such as the posterior mean $\mu_\phi(x)$ can be used as a learned feature representation for downstream tasks such as prediction, clustering, or visualization.

Also: Note that the above VAE assumed the inputs were 

## <a id="proj3"></a> Project 3: Evaluating Numerical Preprocessing Methods for Tabular Representation Learning 
AI Experience Level: **Intermediate  🧩**

Project Type: **Evaluation**

Last updated: **September 28, 2026** 

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
- [Modeling Tabular Data using Conditional GAN](https://arxiv.org/abs/1907.00503) - Focus on Mode-Specific Normalization. Understand why the authors argue that simply scaling a continuous feature may be inadequate sometimes. 
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
- Pick a meaningful train/heldout data split such that we see a large cross-hospital generalization gaps. This can be done by testing different "types" of hospital-to-hospital transfers, and probing which may have the biggest data shifts. 
- Finalize the prediction task(s), eligible hospitals, and train/held-out hospital splits that will be used throughout the project. (Generalization gaps are specific to the prediction task!)
- Train simple prediction models and compare internal (within the same hospital) versus cross-hospital performance.
- Begin exploring & thinking about which features appear important for prediction and which features differ substantially across hospitals.

**Readings:**

- Revisit [Towards global model generalizability: independent cross-site feature evaluation for patient-level risk prediction models using the OHDSI network](https://pmc.ncbi.nlm.nih.gov/articles/PMC11031239/) - Focus on how the authors define cross-site evaluation, how they construct feature sets across sites, and what information from the different sites is used during feature selection.

- [Evaluation of clinical prediction models (part 1): from development to external validation](https://www.bmj.com/content/384/bmj-2023-074819) - Focus on **internal-external cross-validation**.

- Optional: Revisit [Cross-site transportability of an explainable artificial intelligence model for acute kidney injury prediction](https://www.nature.com/articles/s41467-020-19551-w) - Understand how predictive performance and feature importance change across healthcare systems.

**Assignments & Tasks:**
- I have a helper notebook `week2.ipynb` in your project folder to structure this analysis, which will not be due but meant to guide your experiments. 
   -
- Review lecture slides for this week.
- Assignment 1 due Wednesday. Progress Report 2 due Saturday, and should describe your experiments this week. 
- Assignment 2: Related Work assigned. This and all future handins are a group assignment. Read the description on the website. [Nearest neighbor papers listed here.](https://docs.google.com/document/d/10Qe-m0KK5pyykERt7R2zzxDdnpgx3WI7D3OtGlFqDv4/edit?usp=sharing)

---

### Week 3 (10/12 - 10/18): 

**Goals:**
- Understand that there are multiple ways for a feature to be "unstable" across hospitals.
- Implement several candidate measures of feature instability using only the training hospitals.
- Compare instability across different prediction outcomes and feature groups.
- Begin asking: what types of features appear most "stable" vs "unstable"? What measures of instability might actually be useful for predicting poor performance at a future hospital?


**Readings:** I suggest revisiting the following papers for inspiration. What does each paper mean by a "stable" feature? 
- Revisit **StableMate: a statistical method to select stable predictors in omics data** for the   
  distinction between predictive, stable, and environment-specific variables. 

- Revisit **Generalization in Clinical Prediction Models: The Blessing and Curse of Measurement Indicator Variables**  for how "stability" is defined and how variables are grouped. 

- Revisit **Stable Prediction Across Unknown Environments** to understand the high-level motivation for distinguishing stable relationships from correlations that arise because of a particular training environment. You do not need to understand the full method.

**Assignments & Tasks:**
- I have a helper notebook `week3.ipynb` in your project folder to structure this analysis, which will not be due but meant to guide your experiments. 
   * First, define candidate metrics to capture per-variable cross-site "instability" (over the $K$ observed hospitals!).  For example, measuring if for a given variable $X_j$:
      * Marginal distributions of $X_j$ vary across hospitals 
      * Difference in rates of missingness for variable $X_j$
      * Joint distribution ($X_j$ and $X_i$ for all $i$) vary across hospitals 
      * Differences in the conditional outcome distributions $Y\mid X_j$. Caution, however, as this may overlap with measuring predictive utility. 
      * Any other ideas? 
   * Second, find a way to measure the above ideas. For example, comparing marginal distributions of variable $X_j$ can be done thru KS tests across each hospital. Each variable $X_j$ should have on 'instability score' based on the data observed from the $K$ different training hospitals. 
   * Third, experiment with aggregating feature types. Are all lab values more "unstable" than non-lab values? Are some feature types more stable for predicting $Y$ mortality but unstable for another outcome? Compare your results across different outcomes, dataset splits, or models. Look for patterns. 
- Review lecture slides for this week.
- Assignment 2 due Wednesday. Progress Report 3 due Saturday, and should describe your experiments this week. Include all figures and your group's current hypothesis for what makes a feature risky for cross-hospital generalization.
- Assignment 3: Introduction assigned. Read the description on the website.

---

### Week 4 (10/19 - 10/25): 

**Goals:**

- Determine which measures of feature instability from Week 3 appear most relevant to cross-hospital generalization & thus can ideal for generalizable feature selection. 
- Study the tradeoff between a feature's **predictive utility** and its **cross-site instability**.
- Narrow the exploratory results from Week 3 into 1–2 concrete hypotheses about what makes a feature risky for generalization.
- Main question: *Using only data across the training hospitals, can we propose properties of a feature $X_j$ to help us anticipate if including that feature in the model will improve or hurt performance at another unseen hospital?*


**Assignments & Tasks:** Note, no more helper notebooks.
- First, start by creating a measurment for how useful (the *predictive utility*) a feature is in predicting a specific outcome $Y$ for a specific model. Again, this should be computed using only data from the $K$ hospitals. 
- Compare predictive utility measurement to instability / "non-generalizability" measurements. Try creating scatter plots where the x-axis is instability, y-axis is utility, and points are all variables $X_j$.
   - Pay particular attention to features that are: highly predictive and transport well; OR highly predictive but do not transport well. These are the features most relevant to the eventual feature-selection method.
- Use "leave-one-hospital-out" experiments ie. similar to K-fold cross validation. Here you train $K$ models on all but the $i$ th hospital for all $K$ hospitals. This imitates the heldout hospital dataset without peeking into that dataset. You can use this setting to compute if feature utility / instability actually predicts unseen hospital genearlization gap. 
 - Some simple experiment ideas: Compare models using the **most stable** vs. **most unstable** features. What is the genearlization gap? Among features with similar predictive utility, compare more-stable vs. less-stable features.
- Start formulating hypotheses based on the experiments above. For example:
   -  Missingness and healthcare-process variables are disproportionately represented among features that are both highly predictive and highly unstable.
   - This hypothesis should be supported by preliminary evidence from this week's experiments.
- In your progress report, include all figures and hypotheses. You should be able to provide an answer to: Which definition of feature instability currently seems most useful for anticipating poor cross-hospital generalization, and why?
- Review lecture slides for this week.
- Assignment 3 due Wednesday. Progress Report 4 due Saturday, and should describe your experiments this week. 

---

### Week 5 (10/26 - 11/1): 

**Goals:**

- Translate the Week 4 hypothesis into a concrete feature-selection method / ranking of features based on a feature-based score.
- Decide how predictive utility and cross-site instability should be quantified and combined.
- Implement the proposed method using only the training hospitals.
- Verify that the method behaves sensibly within the $K$ training sets using leave-one-out validation before large-scale evaluation + extrapolation to the unseen heldout test data. 

**Assignments & Tasks:** 


- Finalize the primary definition(s) of predictive utility, feature instability, and the proposed combined feature score.

- Implement your proposed stability-aware feature-selection method. Use the score the pick some amount of features that have the highest utility and lowest instability. There are a lot of hyperparameters here for you to try! For example:
  - top `m` features
  - a score threshold
  - varying `lambda` in `Utility - lambda × Instability`
- Use leave-one-training-hospital-out validation of the $K$ hospitals to choose reasonable hyperparameters without using the final held-out hospitals.
- This part is important: Inspect the selected features qualitatively:
  - Which features are consistently ranked as "keep for the model"?
  - Which highly predictive features are removed? Is there any pattern to them? 
  - Do the selections make clinical and statistical sense?
* Produce severeal initial figures showing : the proposed feature score for all features and all feature types; how the selected features changes as the strength of the stability penalty lambda increases; how varying $m$ might change things
* Try for a different outcome $Y$, or a different type of model.
* Review lecture slides for this week.
* Progress Report 5 due Saturday.

---

### Week 6 (11/2- 11/8): 

**Goals:**
- Identify appropriate baselines for comparison, that make the same assumption as our method does. 
- Finalize the evaluation metrics and experimental protocol.
- Ensure that all methods use exactly the same train/validation/heldout splits.
- Freeze the experimental design before running the main evaluation.

**Assignments & Tasks:** 

- Decide on the main feature-selection baselines. Then implement them in code you can easily run.
  - OUR APPROACH 
  - use all features
  - predictive-utility-only selection
  - stability-only selection
  - hospital-identifiability-based selection (ie. features that predict hospital id the least amount)
  - ... other ideas?

- Use the same downstream prediction models for every feature-selection method: eg try logistic regression, XGBoost, maybe an MLP

- Finalize primary evaluation metrics: AUROC, AUPRC, Brier score / calibration, generalization gap. 
- Finalze evaluation strategy: If we have say $M$ unseen hospitals in the heldout dataset, how do we know if a model generalizes well to all $M$? Do we care about worst-case performance? Do we take the median? Which of the above evaluation metrics should we prioritize? 
   - Also: Should we care about a model's performance on an unseen hospital, eg for accuracy `Heldout_Accuracy`, or the generalizaiton gap between `Training_Accurracy - Heldout_Accuracy`?
   - Note, there is no specific answer you should give. Justify all choices you make. 
- Finalize the hospital splits (train/validation/heldout) and prediction outcomes used in the main experiments.

- Decide how many features each method should select so that comparisons are fair. (If we have one baseline select 10 features for predicting Y, but our method selects 20, then our method has more parameters in our model so this is a confounder!)
- Write down the final evaluation protocol before getting results or testing on the heldout hospitals.
- Review lecture slides for this week.
- Progress Report 6 due Saturday.


---

### Week 7 (11/9- 11/15): 

**Goals:**

- Run and compare your proposed feature-selection method / scoring function against all baselines.
- Evaluate whether improvements generalize across multiple unseen hospitals.
- Test whether results are consistent across prediction tasks and model classes.
- Identify where the proposed method succeeds and where it fails.

**Assignments & Tasks:** 
- Run every feature-selection method on the finalized hospital splits.

- Evaluate each selected feature set using all model types (ie. XGBoost ...)

- Repeat the evaluation across:
  - all held-out hospitals
  - multiple outcomes, if available
  - multiple feature-set sizes

- Create the primary results table comparing internal and held-out-site performance.

- Create a figure showing performance across held-out hospitals for each feature-selection strategy.

- Identify the settings where:
  - our approach improves generalization
  - it performs similarly to standard feature selection
  - it performs worse

- Write 2–3 preliminary conclusions supported directly by the main experiments.
- Review lecture slides for this week.
- Progress Report 7 due Saturday. Last one!
- Assignment 4: Evaluation Plan is released. Check the webpage for details. [Note: Assignment 5: Draft Paper due in 2 weeks during Thanksgiving break.]



---
### Week 8 (11/16- 11/22): 

**Goals:**

- Determine whether the main findings are robust to other reasonable analysis choices. This is sometimes called "ablations" in CS research. The goal is to understand which parts of the proposed method are actually responsible for its performance.
- Quantify the tradeoff between internal predictive performance and external generalization.

**Assignments / Project Tasks:**

- Think of any ablations of the proposed method to try. I.e. vary lambda, vary the instaiblity metric by a little bit... Plot the results in a table or figure. In other words, test sensitivity to alternative instability definitions or hyperparameter choices.

- CRUCIAL: Analyze the internal-versus-external performance tradeoff:
  - Vary the number of selected features $m$ and plot the resulting performance curves.
  - Does increasing stability reduce internal performance (i.e. average AUROC on the validation set in the $K$ hospitals)?
  - Is there a region where external performance improves with little internal cost? (look up "pareto curves")

- Summarize which components of the proposed method appear necessary and which provide little additional value.

- Check whether results are driven by:
  - one unusually difficult hospital
  - one prediction task
  - one model class
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
- Understand **why** certain features generalize poorly.
- Finalize the scientific interpretation of the results.
- Finalize all final figures and tables needed for the paper.
- Turn the project into a coherent research paper & research talk. You got this!

**Assignments & Tasks:** 

- Compare instability and selection rates across feature categories, ie. demographics, labs, vitals, diagnoses, procedures, meds, missingness, workflow / hospital variables 
- Identify features that are:
  - highly predictive and stable
  - highly predictive but unstable
  - frequently removed or kept by your proposed method
- Investigate whether specific feature categories disproportionately contribute to large generalization gaps. Why do you think this is the case? 
   - Are there any experiments you can do to test this?
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
