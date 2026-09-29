# Project 1 & 2: Generalization

**PLEASE DO NOT SHARE THIS MATERIAL BEYOND CS197 AT STANFORD UNIVERSITY**


Machine learning (ML) models are often deployed on data that differ from the data used during model training, known as an issue of data distribution shift. Unfortunately, this data shift typically leads to degraded model performance in deployment. 

This problem is particularly important in healthcare applications, when model predictions may mean suboptimal medical treatment. For instance, a clinical prediction model may be developed using data from one set of hospitals, but will later be deployed at a hospital whose patient population, clinical workflows, measurement practices, and treatment patterns differ from those seen during training. In many realistic settings, ML developers may have little or no access to data from the future deployment hospital when the model is developed.

The goal of **domain generalization** is therefore to develop models that perform well on previously unseen sites using only information available from the training sites. 

<!-- The projects are as follows, with the hyperlinks attached:

- [Project 1 (Beginner Level 👍): Benchmarking Generalization Methods for Cross-Hospital Prediction](#proj1)
- [Project 2 (Intermediate Level 🧩): Feature Selection for Cross-Hospital Generalization](#proj2) -->



<!-- ## <a id="proj1"></a> Project 1: Benchmarking Generalization Methods for Cross-Hospital Prediction
AI Experience Level: **Beginner 👍**

Project Type: **Evaluation**

A large literature has proposed different approaches for improving generalization, including changes to data preprocessing, feature representations, and model-training objectives. In this project, you will benchmark how these approaches perform compared to standard empirical risk minimization (ERM) in the context of cross-hospital clinical prediction using the eICU dataset. Specifically, you will train models (i.e., linear regression, decision trees) using data from several hospitals and evaluate the models on hospitals that were completely unseen during model development.

The project has three main goals:
1. **Measure cross-hospital generalization gaps.** Determine what metrics we should care about for measuring generalization. Then determine how much model performance changes when a model is evaluated on an unseen hospital.
2. **Benchmark methods designed to improve generalization.**
Compare against standard ERM.
3. **Understand when and why methods fail.** Analyze why certain generalization approaches may work well, and why others do not or even make performance worse. Can this be attributable to the types of data shift?


### Week 0 (9/21 - 9/27): 
**Onboarding:** 
 - Follow the [instructions provided under "Getting Started"](../README.md) to get setup with the code and data.

**Readings:** 
- [Generalization—a key challenge for responsible AI in patient-facing clinical applications](https://www.nature.com/articles/s41746-024-01127-3)
- [Generalizing to Unseen Domains: A Survey on Domain Generalization](https://arxiv.org/pdf/2103.03097)
- [Healthcare Algorithms Don’t Always Need to Be Generalizable](https://hai.stanford.edu/news/healthcare-algorithms-dont-always-need-be-generalizable)

**Assignments:** 
* See course website for what is due. You will typically have a progress report due Saturday evening and often an assignment due Wednesday morning. This week you have a progress report due Saturday. Since it is your first week, you may not have much to report. 
--->




## <a id="proj2"></a> Project 2: Feature Selection for Cross-Hospital Generalization
AI Experience Level: **Intermediate  🧩**

Project Type: **New Method**

Last updated: **September 28, 2026** 

Clinical prediction models are often trained using features $X$ whose distributions and relationships with an outcome $Y$ vary across hospitals. Some of these features may be highly predictive of $Y$ within the hospitals used for training the models, yet rely on site-specific patient populations, clinical workflows, measurement practices, or treatment patterns. Consider if hospitals collect blood pressure i measurements one way, but then are applied to a different hospital with a different workflow. A model that relies heavily on such features may therefore perform poorly when deployed at a new hospital.

At the same time, variation across hospitals does not necessarily mean that a feature should be removed. A clinically important variable may differ substantially across sites while still containing useful and transportable information about the outcome. In the example above, if we are predicting hypertension, we wouldn't want to remove blood pressure measurements just because they might vary across hospital settings. Removing every feature that exhibits cross-hospital variation could therefore sacrifice both predictive performance and clinical validity.

This project studies the tradeoff between *predictive utility* and *cross-hospital stability*. Suppose we are given data from $K$ observed hospitals during training, and no access to data from the eventual target hospital. Note, the exact formulation in math is the following: Suppose we are given data $(X_1, Y_1) \sim H_1, \ldots (X_K, Y_K) \sim H_K$ sampled from $K$ different hospital environments $H_1 \ldots H_K$, but we do not have data from an unseen hospital $H_{K+1}$. 

Under these assumptions, the central question is:

> **Can heterogeneity across the observed hospitals help us identify which EHR features are likely to hurt generalization to an unseen hospital, while retaining features that are clinically useful despite some cross-site variation?**

The project has four guiding research questions (RQs):

- **RQ1: Which features are predictive but unstable across hospitals?**  
  How should we measure feature instability, and which notions of instability are most informative for predicting future generalization failure?

- **RQ2: Can stability-aware feature selection improve performance on unseen hospitals?**  
  The goal is to develop a simple feature-selection method that considers both predictive utility and cross-site stability. For example:

  ```text
  Feature Xi Score = Predictive Utility - λ × Instability
  ```
   The exact definition of `Instability` will be developed and evaluated during the project.


- **RQ3: What is the tradeoff between internal and external performance?**
      Does removing unstable features improve performance on unseen hospitals at the cost of accuracy within the training hospitals?

- **RQ4: What types of EHR features are most associated with poor generalization?** 
   Are particular feature classes—such as physiologic measurements, procedures, medications, missingness indicators, or healthcare-process variables—systematically more unstable or more harmful to cross-hospital transportability?


---

### Week 0 (9/21 - 9/27): 
**Onboarding:** 
 - Follow the [instructions provided under "Getting Started"](../README.md) to get setup with the code and data.

**Readings:** 
- [Generalization—a key challenge for responsible AI in patient-facing clinical applications](https://www.nature.com/articles/s41746-024-01127-3)
- [Towards global model generalizability: independent cross-site feature evaluation for patient-level risk prediction models using the OHDSI network](https://pmc.ncbi.nlm.nih.gov/articles/PMC11031239/)
- [Stabilizing Variable Selection and Regression
](https://arxiv.org/abs/1911.01850) - This method is very similar what we want to do and there is even [an application on omics data](https://academic.oup.com/nargab/article/6/4/lqae130/7786164). However, it uses some challenging concepts from causal inference for identification of the best variables. Feel free to skim & use AI to help understand.

**Assignments:** 
* See course website for what is due. You will typically have a progress report due Saturday evening and often an assignment due Wednesday morning. This week you have a progress report due Saturday. Since it is your first week, you may not have much to report. 

---

### Week 1 (9/28 - 10/3):

**Goals**: 
* Understand the structure of the eICU dataset, including available features, outcomes, and hospital identifiers.
* Become familiar with several ways prior work has defined or identified unstable features across environments.
* Begin thinking about: why a feature can be predictive within one hospital but unreliable across hospitals. Think about which types of EHR features may be especially vulnerable to cross-hospital instability.



**Readings**: In some of these papers, the methodology may be challenging. Prioritize understanding the motivation and intuition. You can use AI to help you understand the challenging parts. 
- [Learning Optimal Features via Partial Invariance](https://ojs.aaai.org/index.php/AAAI/article/view/25875) - Focus on the main idea: why requiring complete hospital invariance across environments may remove useful predictive information.
- [Generalization in Clinical Prediction Models: The Blessing and Curse of Measurement Indicator Variables](https://pmc.ncbi.nlm.nih.gov/articles/PMC8238368/) - Focus on why variables describing whether or how something was measured can be predictive within a hospital yet fail to generalize across hospitals.
- [Stable Prediction across Unknown Environments](https://dl.acm.org/doi/10.1145/3219819.3220082) - First, understand what does "un/stable prediction" mean? Then on the distinction between stable and unstable relationships and why ordinary prediction models may exploit variable relationships that will then fail under distribution shift, i.e., on different hospital environments.
- [Cross-site transportability of an explainable artificial intelligence model for acute kidney injury prediction](https://www.nature.com/articles/s41467-020-19551-w) - Focus on how feature importance and predictive performance vary across healthcare systems, and what this suggests about cross-site transportability.

<!-- A Theoretical Analysis on Independence-driven Importance Weighting for Covariate-shift Generalization -->


**Assignments & Tasks**: 
- Review lecture slides for this week.
- By Wednesday, assuming you applied for it last week as expected, you should have been granted access by PhysioNet to the full eICU dataset. If you have not received an email by then, then email me and include the date you applied.
- After you get data access, follow the rest of the instructions on ``Getting Started`` to download the data.
- Then you should run the notebook `week1_explore_eicu_data.ipynb`. This achieves two purposes: 
   - First, you should have a good understanding of the underlying dataset, what features are available, how they are reprsented, and what clinical labels exist. This will help you a lot as the quarter progresses. 
   - Second, this notebook will generate the dataframe `../data/clean_dataset.parquet` which you will need for the project! 
- For Assignment 1 (due next week): This should be done solo. All other assignments will be done in your group. 
   - Please reread the paper from Week 0 [Towards global model generalizability: independent cross-site feature evaluation for patient-level risk prediction models using the OHDSI network](https://pmc.ncbi.nlm.nih.gov/articles/PMC11031239/) (for Part A: Read a Paper)
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
   * Finalize the main prediction outcome(s) $Y$ that your group will use, e.g., mortality, heart attack at 48 hours, etc. Review the week1 notebook's description of timing of outcomes.Also finalize what features $X$ you want to use to predict $Y$.  
   - Identify hospitals with sufficient sample size and outcome prevalence for reliable evaluation.
   - Define your data split. You should do this several times until you find a large gap. For example, split on numer of bed counts, hospital's region in the US, etc.
   - Training data = A set of $K$ hospitals that can be used during model development. You should have a pretty good sample size overall (i.e. > 10k patients across $K$ hospitals)
   - Heldout data = Several (but <10) **held-out hospitals that should not be used for feature selection or model tuning**, except for this week. 
   - For each outcome $Y$, train a simple baseline model using the $K$ training hospitals. For example, XGBoost or logistic regression.
   - Evaluate performance within the training distribution and separately on each held-out hospital.
   - Create at least one figure showing the **generalization gap across hospitals**, i.e., Internal training data performance minus the Held-Out hospital performance.
   - Begin exploring possible explanations for the largest gaps. Do we think the biggest gap will come from:
      - differences in patient populations
      - outcome prevalence
      - feature distributions
      - missingness rates
      - hospital-specific feature importance
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