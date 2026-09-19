# Project 1 & 2: Generalization

**PLEASE DO NOT SHARE THIS MATERIAL BEYOND CS197 AT STANFORD UNIVERSITY**


Machine learning (ML) models are often deployed on data that differ from the data used during model training, known as an issue of data distribution shift. Unfortunately, this data shift typically leads to degraded model performance in deployment. 

This problem is particularly important in healthcare applications, when model predictions may mean suboptimal medical treatment. For instance, a clinical prediction model may be developed using data from one set of hospitals, but will later be deployed at a hospital whose patient population, clinical workflows, measurement practices, and treatment patterns differ from those seen during training. In many realistic settings, ML developers may have little or no access to data from the future deployment hospital when the model is developed.

The goal of **domain generalization** is therefore to develop models that perform well on previously unseen sites using only information available from the training sites. 

The projects are as follows, with the hyperlinks attached:

- [Project 1 (Beginner Level): Benchmarking Generalization Methods for Cross-Hospital Prediction](#proj1)
- [Project 2 (Intermediate Level): Feature Selection](#proj2)



## <a id="proj1"></a> Project 1: Benchmarking Generalization Methods for Cross-Hospital Prediction
AI Experience Level: **Beginner**

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


### Weeks 1-10: TBD 

<br>





<!-- ### Week 1: 
**Onboarding:** 
 - Follow the [instructions provided under "Getting Started"](../README.md) to get setup with the code and data.
 - **Important:** You will need to generate and save the full dataset using the notebook from Week 2. 


**Readings:**
- [In Search of Lost Domain Generalization](https://arxiv.org/pdf/2007.01434) - another benchmark paper
- [Benchmarking Distribution Shift in Tabular Data with TableShift
](https://proceedings.neurips.cc/paper_files/paper/2023/file/a76a757ed479a1e6a5f8134bea492f83-Paper-Datasets_and_Benchmarks.pdf) - another benchmark paper
- [Distributionally Robust Neural Networks for Group Shifts: On the Importance of Regularization for Worst-Case Generalization](https://arxiv.org/abs/1911.08731) - methods paper we will try = group DRO
- []()


**Notebook:** 
- Walk through the notebook `week1_explore_eicu_data.ipynb` to become familiar with the eICU dataset. 

*For Assignment 1:*  Please read [An Empirical Characterization...]((http:/s/pmc.ncbi.nlm.nih.gov/articles/PMC7871979/)) (for Part A: Read a Paper) and turn in your outputs of section *3. Section Starter: Now it's your turn!* in `week1_explore_eicu_data.ipynb` as a pdf (as this assignment's Part 2: Section Starter Task).

*For Progress Report 1:* Meet with your project group and submit what you all want to accomplish for Week 3. [See the website](https://web.stanford.edu/class/cs197/assignments/project.html#progress-reports) for how we expect project reports to be structured. -->
<!-- 
### Week 3: 
**Onboarding:** 
- By Wednesday, you should have been granted access by PhysioNet to the full eICU dataset. Email me if you have not received an email by then. 
- You will need to generate and save the full dataset using the notebook from Week 2. 

**Additional Readings:** In conjunction with the readings for Assignment 2. 

- [Dissecting racial bias in an algorithm used to manage the health of populations](https://www.science.org/doi/10.1126/science.aax2342)
- [Ensuring Fairness in Machine Learning to Advance Health Equity](https://www.acpjournals.org/doi/epdf/10.7326/M18-1990)
- [Algorithmic fairness in computational medicine](https://pmc.ncbi.nlm.nih.gov/articles/PMC9463525/)

**Video:**
- Watch [this video](https://www.youtube.com/watch?v=MzuoWAk9_AQ) from 21:00 to 1:21:00

**Notebook:** 
- Walk through and complete the notebook `proj1/week2_fairness_evaluation.ipynb` to become familiar with algorithmic fairness calculations.

*[For Assignment 2](https://web.stanford.edu/class/cs197/assignments/project.html#related-work) (Due April 16):* You will explore related work in this field. Please see the "nearest-neighbor" papers for this project [here](https://docs.google.com/document/d/10Qe-m0KK5pyykERt7R2zzxDdnpgx3WI7D3OtGlFqDv4/edit?usp=sharing). 

*For Progress Report 2 (Due April 19):* Meet with your project group and submit what you all want to accomplish for Week 4. [See the website](https://web.stanford.edu/class/cs197/assignments/project.html#progress-reports) for how we expect project reports to be structured.


### Week 4 (April 20 - 26): Is missingness informative?
**Goal:**
Is missingness random, or informative, in the eICU dataset?

**Readings:**
- [Imputation Strategies Under Clinical Presence: Impact on Algorithmic Fairness](https://proceedings.mlr.press/v193/jeanselme22a/jeanselme22a.pdf) - read in-depth
- [Fairness in Missing Data Imputation](https://arxiv.org/pdf/2110.12002) - read in-depth
- [Exploring the Inequitable Impact of Data Missingness on Fairness in Machine Learning](https://ieeexplore.ieee.org/document/10920480) - skim
- [Adapting Fairness Interventions to Missing Values](https://arxiv.org/pdf/2305.19429) - skim
- [Missing data and multiple imputation in clinical epidemiological research](https://pmc.ncbi.nlm.nih.gov/articles/PMC5358992/) - skim


**Tasks**: 

(Note: The `week3_fairness_tradeoffs.ipynb` notebook is now entirely optional / not required.)

1. Examine the distribution of missing features (missingness frequency per feature and per patient, and then an overall histogram of missingness frequency across all features and all patients). Plot using seaborn to visualize your findings. 
2. Analyze how missingness varies across: sex, ethnicity, age, hospital characteristics (e.g., hospitalid, region, bed count). Do certain groups systematically have more or less complete data? Use a statistical test (i.e., two-sampled KS test, see [here](https://pmc.ncbi.nlm.nih.gov/articles/PMC8327789/) if you are unfamiliar with statistical tests) and plot using seaborn to determine the answer. Furthermore, do missing values correlate across the features themselves? 
3. Construct a "missingness attribute". For example, you can define patient groups based on if they have low / medium / high rates of feature missingness, you can cluster using KNN based on binary missingness masks. Visualize and interpret these groups.
4. Train a classifier model using only missingness indicators (binary features) to predict mortality. Use interpretability tools (i.e., coefficients (for linear classifiers), or SHAP values (for tree-based including XGB classifiers)). Identify which missing features are most predictive and see if this makes sense given the task at hand.
5. Based on your findings, argue if missingness is random (uninformative), or structured (reflecting clinical processes, access, or data collection differences). Relate this to MAR, MCAR, and MNAR (you should have learned about these in your assigned readings!). 
6. Thoroughly discuss how missingness may impact (good or bad) the "standard" fairness evaluation framework.  

**Deliverables:**
* Progress Report 3 (Due April 26) - a minimum 2-page writeup plus a notebook of all the completed tasks above. As a reminder: $\checkmark+$ = 100% indicates you went above and beyond; $\checkmark$ = 95% indicates basic completeness. 
* Introduction (Due April 23) - see website.

### Week 5 (April 27 - May 3): Can missingness be a sensitive attribute?

**Goal**:
Can missingness itself be treated as a "sensitive attribute" for evaluating fairness, as an alternative or complement to race and sex?

**Tasks**: 
1. Decide on a way to categorize missingness levels (i.e., using your "missingness attribute" from Week 4). For a classifier trained to predict mortality, pick one imputaiton strategy (we will analyze this strategy further next week), and analyze the "standard" four metrics of fairness (see A2 above) using the following 3 sensitive attributes: (1) missingness category, (2) race, and (3) sex. Which missingness group perform the best and the worst? How does this performance compare to the best / worst groups defined by race and sex? 
2. Construct intersectional subgroups: e.g., (race × missingness level), (sex × missingness level), (sex x race x missingness level). Evaluate fairness performance across these groups, and report the bootstrapped variance of these fairness metrics (so we can see how small sample size affects the consistency of fairness metrics). Summarize your findings. 
3. Repeat these experiments where we look at a different "missingness attribute" based on a different set of feature missingness. For example, if your original category was based on missingness of ALL variables, evaluate over a new cateogry of missingness of JUST lab variables. 
4. Reflect: Should missingness be considered a fairness-relevant attribute? Why or why not? 

**Deliverables:**
* Progress Report 4 (Due May 3) - a minimum 2-page writeup plus a notebook of all the completed tasks above 
* (Optional) Related Works - based on this project direction, refine your current related works section to focus in on algorihtmic fairness specifically with respect to missing data. You will need to do this eventually, so might be a good idea to work on that this week. 

### Week 6 (May 4 - 10): Does missing data handling affect fairness evaluations?

**Goal**:
Do different missing data handling strategies lead to different fairness conclusions?  

**Tasks**:
1. Define a fairness evaluation protocol (i.e., fix a model, split, and evaluation metrics). 
2. Implement at least three strategies for handling missing data, such as (a) complete-case analysis - dropping all patients who have feature/s X missing, (b) simple imputation (mean/median), (c) using missingness indicators, (d) MICE or other advanced method. For each, track dataset size (how many patients remain), and track feature distributions (before vs after).
3. Measure fairness using the four standard metrics across all group definitions (i.e., race, sex, age, missingness attribute, and intersecitonal groups). Compare how the different strategies for handling missing data affect the fairness results. 
4. Introduce mild synthetic missingness (e.g., drop 10–20% of values randomly = MCAR or based on some attribute = MAR). Re-run preprocessing pipeline and train on this synthetically missing data, but when you evaluate, evaluate on the original dataset without the 10-20% dropped. Check whether fairness conclusions remain stable.
5. Reflect on how missing data preprocessing affected the fairness conclusions.  

**Deliverables:**
* Progress Report 5 (Due May 10) - a minimum 2-page writeup plus a notebook of all the completed tasks above 

### Week 7 (May 11 - 17): Final experiments on missingness intersectional fairness

This is our last week of experiments before we start writing the final paper. So far, we have explored several related threads around missing data and algorithmic fairness. In Week 5, we observed that fairness gaps may look different when we stratify patients not only by race or sex, but also by how much data is missing for them. The goal this week is to formalize those analyses for a final paper thesis.

We want to ask whether missingness itself should be treated as an additional axis of fairness. In other words, we want to know whether patients with similar race or sex labels but very different missingness patterns experience different model performance and fairness outcomes.

*Note: this todo list looks long, but it is mostly just cleaning up and repeating experiments from prior weeks.*

1. First, we need to make sure the general modeling setup is clear and reproducible (this was similar to task 1 last week, but I am making it more clear what needs to be done here.)
   * First we will define two prediction tasks: (1) Mortality prediction and (2) Future troke prediction (you can also pick another health outcome; just make sure you are using *future* 24 hr - 48 hr ICD10 codes, not the diagnoses within the first 24 hrs.) For each task, use approximately 15-100 features $X$ to predict the two outcomes $Y$. You should not include race or hospital beds in the prediction task (sex is ok). 
   * For the paper appendix, you should a table listing: the features used for each prediction task, missingness distribution of the features (as well as the top and bottom 10 missing features). Also explictly document the missingness rates of all sensitive attributes and all outcomes.
   * For this first baseline analysis, use either mean or median imputation for numerical variables, and use the most frequent category. For now, also use simple handling of missing values in Y and sensitive attributes A.
   * Train models for both prediction tasks using either: LogisticRegressionCV (sklearn) or XGBoostClassifier. Use a train / validation split. All fairness metrics should be reported on the validation set.
   * For each task, report average AUPRC plus standard fairness metric gaps (see below) across the following 3 sensitive attributes: race, sex, number of hospital beds, treated as a proxy for hospital capacity.
   * The fairness metrics should include calibration gap, and equalized odds gap (TPR gap). (Ask yourself: why do we care about these two metrics specifically here?) For each metric, report the gap as the difference between the maximum group value and the minimum group value. For example, for equalized odds, report the fairnes gap $g_{TPR}$ as $$g_{TPR} = \max_{a \in A} TPR(a) - \min_{a \in A} TPR(a)$$ for sensitive attribute/s based groups $A$. 
      
   * The main result from this section should be a table with: 2 prediction tasks x 2 fairness metrics x 3 sensitive attributes.

      You should also report the number of groups for each sensitive attribute, the group with the minimum metric value, and the group with the maximum metric value.

      This table gives us the baseline fairness audit that a standard algorithmic fairness paper might report.
2. Next, we will extend the fairness audit by treating missingness as an additional fairness-relevant attribute. Instead of only evaluating fairness across race, sex, and hospital bed count, we will evaluate fairness across intersectional groups of the form: sensitive attribute x missingness group. We previously explored this in Week 5. 

   * For the final project, we will only define missingness using variables from the following data sources: patient data, hospital data, labs.
   * First, recompute the missingness percentages from Week 5. Then define binned missingness groups as low, medium, and high missingness based on tertiles of their missingness percentage.
   * Second, define missingness clusters using the missingness matrix R, where each row is a patient and each column corresponds to whether a variable is missing. Apply PCA to the missingness matrix. Cluster patients using KMeans on the top principal components. Try multiple values of K and use an elbow plot or silhouette score to pick a reasonable number of clusters.
   * Then recompute the standard fairness table, but now using the 6 new intersectional missingness groups: race x binned, race x KMeans missingness cluster,... etc. 
   * Are any gaps larger? Do we learn any new unfairness patterns across race (i.e. race x missingess has higher gaps than just race alone) We are especially interested in cases where the race-only or sex-only fairness gap appears small, but the race x missingness or sex x missingness gap is large. Those cases would support the argument that traditional fairness audits can miss clinically meaningful model failures related to missing data.


3. Next, we will evaluate whether the observed fairness gaps are stable across different imputation methods. We have results on median / mean imputation. Repeat all evaluation above  MICE imputation. Note: The imputation procedure (mean imputation, MICE, etc. ) should be fit only on the training data and then applied to the validation data. 

 4. In the prior experiments, missingness mainly enters the analysis through imputation and group stratification. In this final experiment, we will explicitly add missingness indicators as model features and compare the results to models that only use imputed clinical values. For each task and for mean imputation strategy, double the features of $X$ where each feature $X_j$ also now has the missingness indictator $R_j = 1$ if feature $X_j$ was missing before imputation and $R_j = 0$ otherwise. Compare both the average AUPRC performance plus the two standard fairness metric gaps. Based on these results, we can assess: 
      * If adding missingness indicators improves overall performance but worsens fairness gaps, this suggests the model may be exploiting missingness patterns in a way that benefits some groups more than others.
      * If adding missingness indicators barely changes performance or fairness, this suggests that the original imputed values may already capture most of the relevant missingness information.
      * ... etc. 

### Week 8 (May 18–24): Collecting Results, Tables, and Figures

This week, we should move from trying new analyses to organizing the evidence we already have. As you write the results section, start discussing with your group on what the narrative of the paper is, and how to build the results section to support this narrative. 

For example, the reults should include clean tables / figures of: 
   * A standard fairness audit table.
   * A missingness-aware (intersectional groups) fairness audit table.
   *  An imputation comparison table.
   *  A missingness-indicator comparison table.

In addition to the tables, you should also make one or two clear plots summarizing the most important fairness gap comparisons from above. For example: 
- bar plots comparing standard fairness gaps versus missingness-aware fairness gaps,
- heatmaps of fairness gaps by task, metric, and group definition,
- (optional) missingness cluster interpretation plots - what does each missingness cluster represent in terms of patient cohorts? This might be helpful for justifying your conclusions
- plots showing how fairness gaps change across imputation methods,
- plots comparing imputed-values-only models to imputed-values-plus-missingness-indicator models.

Every possible plot will not be included in the main paper, but we would like to organize all results so that we can decide which findings are central, which belong in the appendix, and which should be dropped.

 Also write a short outline of the Results section: What is the main story? Does missingness reveal fairness failures that are hidden by standard demographic audits? Are the results consistent across tasks and imputation methods?


### Week 9 (May 25–31): Writing the First Full Draft

This week, the goal is to turn the results into a first full draft of the paper. Start with the Methods and Results sections, since these should be directly grounded in the experiments. The Methods section should clearly explain the prediction tasks, feature sets, imputation methods, fairness metrics, missingness group definitions, and missingness-indicator experiment. The Results section should walk the reader through the standard fairness audit, the missingness-aware fairness audit, the imputation comparison, and the model comparison with versus without missingness indicators.

After Methods and Results are drafted, work on the Introduction and framing. The paper should motivate the problem as follows: standard fairness audits often focus on demographic groups such as race and sex, but in clinical data, missingness patterns may reflect differences in measurement, access, severity, hospital practice, or data quality. Our project asks whether missingness should be treated as an additional fairness-relevant attribute, and whether ignoring it can hide model failures. By the end of the week, please have a complete rough draft with all major sections: Introduction, Related Work if applicable, Methods, Experimental Setup, Results, Discussion, and Limitations.


### Week 10 (June 1–7): Final Writing, Polishing, and Presentation

This week, the goal is to finish the paper and prepare the final presentation. 
<!-- Focus on making the writing clear, tightening the main claim, and making sure every claim is supported by a table, figure, or concrete result. The final paper should clearly explain what standard fairness audits show, what additional information is revealed by missingness-aware groups, and how imputation or missingness indicators change the conclusions.

Please also polish the figures, table captions, and appendix materials. The appendix should include feature lists, missingness rates, cohort details, and any additional fairness tables that are too large for the main paper. The final presentation should tell the same story as the paper in a simpler form: what question we asked, why missingness matters for fairness, what experiments we ran, what we found, and what the limitations are. --> 







## <a id="proj2"></a> Project 2: Feature Selection
AI Experience Level: **Intermediate**

Project Type: **New Method**

Motivation

Clinical prediction models often use features $X$ whose distributions and relationships with outcomes $Y\mid X$ differ across hospitals. Although some of these features may be relevant or correlated with the outcome, and thus improve performance, they might also make the model less reliable at a new hospital. Thus, simply removing features that vary across hospitals -- which might allow for a more generalizable model -- can also remove important clinical information.

This project studies the tradeoff between feature predictive utility and cross-site stability. Specifically, if we have data from $K$ hospitals during training, how can we use heterogeneity across the observed hospitals to identify features that are likely to generalize well to a new, unseen hospital?

The project has four guiding research questions (RQ):
* RQ1: Which features are predictive but unstable across hospitals?
<!-- 
For each feature, characterize:

predictive utility for the outcome
variation in its relationship with the outcome across hospitals
variation in its distribution across hospitals
ability to predict hospital identity -->

* RQ2: Can stability-aware feature selection improve unseen-site performance?The goal here is to develop a simple feature-selection method that considers both utility and cross-site stability, for example: ```Feature X_i Score = Predictive Utility - λ × Instability```
<!-- 
Compare against:

all features
standard predictive feature selection / LASSO
stability-only feature selection
hospital-identity-based feature removal -->

* RQ3: What is the internal vs external performance tradeoff?

<!-- Determine whether removing unstable features:

decreases performance within training hospitals
improves performance on unseen hospitals
produces a useful tradeoff between the two -->

* RQ4: What types of EHR features are most unstable? That is, are there patterns in what features makes a prediction model generalize poorly? 

<!-- Compare feature categories such as:

clinical / physiologic measurements
demographics
diagnoses
medications
procedures
missingness indicators
measurement / workflow variables -->

<!-- Ask whether particular categories disproportionately contribute to cross-site performance degradation.

Validation

Use leave-one-hospital-out experiments within the training hospitals to develop the feature-selection method.

After the method is finalized, evaluate it on completely held-out hospitals that were never used for feature selection or tuning. -->

<!-- Expected Outcome

The goal is to understand:

Feature Properties -> Cross-Site Instability -> Generalization

A successful project should produce:

a measure of feature instability,
a simple stability-aware feature-selection strategy,
comparison against standard feature-selection baselines, and
an analysis of which EHR feature types are most associated with poor transportability. -->


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

### Weeks 1-10: TBD 

</br>
<!-- StableMate: a statistical method to select stable predictors in omics data -->
<!-- A Theoretical Analysis on Independence-driven Importance Weighting for Covariate-shift Generalization -->
<!-- - [Cross-site transportability of an explainable artificial intelligence model for acute kidney injury prediction](Cross-site transportability of an explainable artificial intelligence model for acute kidney injury prediction) -->
