# CS197 Project: Exploring Biases in Healthcare Data
**Course Staff:** Kara Liu (contact: karaliu@stanford.edu)

**PLEASE DO NOT SHARE THIS MATERIAL BEYOND CS197/CS195 AT STANFORD UNIVERSITY**

In this project you will be explore how to detect and mitigate biases $^{[0]}$ present in real-world tabular $^{[1]}$ datasets. Real-world data, particularly in healthcare, are a rich source of information in research that can help us understand trends, test new methods, and support real people like clinicians and patients. However, this data is often messy and imperfect. The common phrase "garbage in, garbage out" emphasizes how important data quality is to learn useful and accurate machine learning (ML) models. If the data is incomplete or noisy, our models may draw wrong conclusions and can even lead to unfair outcomes.

In this project, we will be working with the [eICU Collaborative Research Database (eICU-CRD)](https://eicu-crd.mit.edu/), a large, multi-center critical care dataset containing de-identified patient data from intensive care units (ICUs) across the United States. 

$^{[0]}$ Yes, "bias" is a loaded term. We will clarify what we mean by this later.

$^{[1]}$  Tabular means any dataset that contains a mixture of both discrete (country of origin, has hypertension, ...) and continuous (age, height, ....) variables with separate columns.

---

## A. Course Structure & Timeline

This 10-week project is divided into two phases:

* **Weeks 1-2: Onboarding.** Everyone must apply for access to the data and will get familiar with real-world data through an interactive notebook
* **Weeks 3-10: Specialized Project Track.** In Weeks 3 and 4, each project will provide its own specialized curriculum through interactive notebooks and suggested readings. In Weeks 5-10, you will choose a specialized project to own for the rest of the quarter, culminating in a final presentation and report.

The projects are as follows, with the hyperlinks attached:

- [Project 1: Fair ML in healthcare](#c-project-1-fair-ml-in-healthcare)
- [Project 2: Improving selection bias with group DRO](#d-project-2-improving-selection-bias-with-group-dro)

## B. Getting started 

### Data Access
(IMPORTANT) Please complete this step **as soon as possible**, as it will take a bit of time.

1. First, you must request approval for the eICU dataset. To do so, follow the instructions [here](https://eicu-crd.mit.edu/gettingstarted/access/) which consists of three steps: 
    * Complete required CITI training (needed to work with patient data, even if de-identified; takes ~30 minutes  - 2 hours)
    * Register for an account on PhysioNet
    * Submit an application to PhysioNet for access to eICU data. *Note: When it asks for your supervisor name and email, you can give my contact info: Kara Liu, karaliu@stanford.edu*
2. Data acess will take about 1-5 days. If it has been over 7 days since you submitted your PhysioNet application and you have yet to hear back, please email me. 
3. While you wait for access, you can explore the [**demo** dataset](https://physionet.org/content/eicu-crd-demo/2.0.1/) which consists of about ~2,000 patients subsampled from the full ~200,000. You can download it via the terminal: 
    ```
    wget -r -N -c -np https://physionet.org/files/eicu-crd-demo/2.0.1/
    ```
    You do not need to unzip the individual files, we will be reading the `*.csv.gz` files directly. 
4. Once you are granted access to the full eICU dataset (expected in Week 2), you can navigate to the [full dataset's page](https://physionet.org/content/eicu-crd/2.0/) to start downloading the full eICU dataset. You can download via the terminal command:
    ```
    wget -r -N -c -np --user {YOUR_PHYSIONET_USERNAME} --ask-password https://physionet.org/files/eicu-crd/2.0/
    ```
    which will prompt you for your PhysioNet password before starting the download. Note, downloading may take several hours (it took me about ~7) as some of the files are quite large. 
5. After downloading, your folder (for me, it downloaded to the path `./physionet.org/files/eicu-crd/2.0/`) should look something like
    ```
    ./physionet.org/files/eicu-crd/2.0/
    | -- patient.csv.gz
    | -- lab.csv.gz
    | -- hospital.csv.gz
    | -- ...
    ```

Congratulations! You now have access to a real-world clinical dataset.

If this process felt slower or more involved than expected, that’s completely normal. Working with real-world data often includes several time-consuming steps, but the reward is working with data that contains real patient information. This tradeoff is all part of the reserach process.


### Setting up the environment

1. If you haven't already, install [Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/main) or Anaconda, which will help manage your project's software packages.

2. (Optional) Open your terminal and run the following commands. This step is optional but we recommend it as it is good practice to always create a fresh environment to ensure your dependencies don't conflict with other projects.  
    ```
    # Create an environment named 'cs197' with Python 3.12
    conda create --name cs197 python=3.12

    # Activate the new environment
    conda activate cs197
    ```
3. After downloading this repository (i.e. using `git clone`), cd into it and install all necessary package requirements. 
    ```
    cd cs197-bias # if you are not already there 
    pip install -r requirements.txt
    ```

3. Test that you can open the first notebook `week1_explore_eicu_data.ipynb` by launching (where if applicable, make sure you are in the correct conda environment): 
    ```
    jupyter notebook
    ```


## C. Project 1: Fair ML in healthcare


Consider the "standard" algorithmic fairness framework as defined by three key features:
* **A1:** Fairness is evaluated with respect to a small set of predefined sensitive attributes, typically race and sex.
* **A2:** Fairness is measured using a fixed set of standard metrics, such as demographic parity, equal opportunity, equalized odds, and calibration.
* **A3:** Missing data is treated as a preprocessing issue to be handled before evaluating mordel performance - typically, either by complete-case analysis (dropping patients with missing data) or by imputation. 

The research project for the remainder of the quarter is: *How can we improve this standard framework to better account for the information and structure contained in missingness?* Rather than treating missing data as something to "fix", this project investigates whether missingness itself reveals important patterns useable for constructing a better algorithmic fairness framework. 

This project can be broken down into several guiding questions:
* **Q1:** How informative is missingness in the eICU dataset?
* **Q2:** Where (and how) does the standard fairness framework break under missingness?
* **Q3:** Can we extend the fairness framework to incorporate missingness?

The (revised) schedule for the rest of the quarter is as follows, where each weeks is framed by the central questions your group will try to answer that week: 
* **Week 4**: Is missingness random, or informative, in the eICU dataset? 
* **Week 5**: Can missingness itself be a "sensitive attribute" to evaluate fairness over (an alternative to race / sex from A1)?
* **Week 6**: How do different missing data handling strategies (A3) affect fairness metrics?  
* **Week 7**: How can we extend the "standard" fairness frameworks listed in A1, A2, or A3 to better handle or incorporate missingness? 
* **Week 8-10**: Experiments / writing 

### Weeks 1 - 2: 
**Onboarding:** 

 - Follow the [instructions provided in section B.](#b-getting-started) to get setup with the code and data.

**Readings:**
- [An Empirical Characterization of Fair Machine Learning For Clinical Risk Prediction](http:/s/pmc.ncbi.nlm.nih.gov/articles/PMC7871979/): Please read and use for Assignment 1.
- [A brief review on algorithmic fairness](https://link.springer.com/article/10.1007/s44176-022-00006-z): Review paper, feel free to skim parts that are already familiar to you. 

**Additional Readings:** Review papers on the general field of bias in healthcare data. Optional but highly encouraged. 
- [Potential Biases in Machine Learning Algorithms Using Electronic Health Record Data](https://pmc.ncbi.nlm.nih.gov/articles/PMC6347576/)
- [Unmasking bias in artificial intelligence: a systematic review of bias detection and mitigation strategies in electronic health record-based models](https://pubmed.ncbi.nlm.nih.gov/38520723/)
- [Bias in medical AI: Implications for clinical decision-making](https://pmc.ncbi.nlm.nih.gov/articles/PMC11542778/)
- [Lessons and tips for designing a machine learning study using EHR data](https://pmc.ncbi.nlm.nih.gov/articles/PMC8057454/)

**Notebook:** 
- Walk through the notebook `week1_explore_eicu_data.ipynb` to become familiar with the eICU dataset. 

*For Assignment 1 (Due April 12):*  Please read [An Empirical Characterization...]((http:/s/pmc.ncbi.nlm.nih.gov/articles/PMC7871979/)) (for Part A: Read a Paper) and turn in your outputs of section *3. Section Starter: Now it's your turn!* in `week1_explore_eicu_data.ipynb` as a pdf (as this assignment's Part 2: Section Starter Task).

*For Progress Report 1 (Due April 12):* Meet with your project group and submit what you all want to accomplish for Week 3. [See the website](https://web.stanford.edu/class/cs197/assignments/project.html#progress-reports) for how we expect project reports to be structured.

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

## D. Project 2: Selection-Aware Robust Learning for Cross-Hospital Generalization

In this project, we aim to develop clinical prediction models that generalize across hospitals. For example, we may want to predict mortality risk on patients across the U.S. using EHR data as input, but in practice we only observe data from a subset of hospitals. Models trained on this biased data subsample often fail when deployed to new hospitals.

A key challenge towards model generalization is the presence of *selection bias and distribution shift* in the training data. Consider that patients in EHR data are not representative of the underlying population (e.g., patients with data often have above average healthcare access). Furthermore, hospitals differ in patient mix, measurement practices, coding, and missingness patterns. As a result, the observed training distribution differs from the (unknown) deployment distribution (i.e., a new hospital) both in terms of hospital environments and population composition. This leads to models that might fail to perform in new clinical environments. 

A common approach to improving robustness is **Group Distributionally Robust Optimization (gDRO)**, which optimizes worst-case model performance over predefined groups (e.g., race, sex, or hospital). However, this requires the causes of the data distribution shift to be fully captured by these observed group labels. In many settings (including hospitals), this assumption is likely violated, as groups defined by hospital ID are likely only a proxy for other sources of shift.

The key question this project seeks to answer is: *How can we extend Group DRO to account for selection bias and improve generalization to unseen hospital environments?* We can further break down this project into several key questions:

* **Q1:** Does standard Group DRO improve generalization for cross-site (hospital) settings? (this would motivate the "bit-flip")
* **Q2:** Can we learn a "latent" driver of dataset shift? 
   - Are hospital labels sufficient to capture distribution shift, or are they merely a proxy? 
   - Are there other variables we observe that drive data shift? 
* **Q3:** Does soft group membership perform better than hard  group membership?

<!-- The (revised) schedule for the rest of the quarter is as follows, where each weeks is framed by the central questions your group will try to answer that week:

* **Week 4:** How does model performance change across hospitals? What evidence is there of selection bias? And does regular group DRO help / hinder this performance?
* **Week 5:** Does assigning a soft group membbership using proxy selection variables improve group DRO?
* **Week 6:** Does using uncertainity set reweighting help? 
* **Week 7:** ...?
* **Week 8-10:** Refinement, analysis, ablation studies -->

---
$^{[1]}$: Many different fields have names for this phenomenon. It may also be called *distribution shift*, *data shift*, *dataset bias*, *sample* selection bias, and in some cases, *covariate shift*. In ML, you might see that the field of domain adaptation is relevant for selection bias. The term *selection bias* I am using has its origins in the field of causal inference.



### Weeks 1-2: 
**Onboarding:** 

 - Follow the [instructions provided in section B.](#b-getting-started) to get setup with the code and data.

**Readings:**
- [Sample Selection Bias in Machine Learning for Healthcare](https://dl.acm.org/doi/pdf/10.1145/3761822)
- (News article) [Research suggests Epic Sepsis Model is lacking in predictive power](https://www.healthcareitnews.com/news/research-suggests-epic-sepsis-model-lacking-predictive-power)
- [A Unified Framework on Generalizability of Clinical Prediction Models](https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2022.872720/): Feel free to skim. 



**Additional Readings:** Review papers on the general field of bias in healthcare data. Optional but highly encouraged. 
- [Potential Biases in Machine Learning Algorithms Using Electronic Health Record Data](https://pmc.ncbi.nlm.nih.gov/articles/PMC6347576/)
- [Unmasking bias in artificial intelligence: a systematic review of bias detection and mitigation strategies in electronic health record-based models](https://pubmed.ncbi.nlm.nih.gov/38520723/)
- [Bias in medical AI: Implications for clinical decision-making](https://pmc.ncbi.nlm.nih.gov/articles/PMC11542778/)
- [Lessons and tips for designing a machine learning study using EHR data](https://pmc.ncbi.nlm.nih.gov/articles/PMC8057454/)


**Notebook:** 
- Walk through the notebook `week1_explore_eicu_data.ipynb` to become familiar with the eICU dataset. 

*For Assignment 1 (Due April 12):*  Please read [Sample Selection Bias in Machine Learning for Healthcare](https://dl.acm.org/doi/pdf/10.1145/3761822) (for Part A: Read a Paper) and turn in your outputs of section *3. Section Starter: Now it's your turn!* in `week1_explore_eicu_data.ipynb` as a pdf (as this assignment's Part 2: Section Starter Task).

*For Progress Report 1 (Due April 12):* Meet with your project group and submit what you all want to accomplish for Week 3. [See the website](https://web.stanford.edu/class/cs197/assignments/project.html#progress-reports) for how we expect project reports to be structured.

### Week 3: 
**Onboarding:** 
- By Wednesday, you should have been granted access by PhysioNet to the full eICU dataset. Email me if you have not received an email by then. 
- You will need to generate and save the full dataset using the notebook from Week 2. 

**Additional Readings:** In conjunction with the readings for Assignment 2. 
- [A Review of Domain Adaptation without Target Labels](https://pubmed.ncbi.nlm.nih.gov/31603771/): Feel free to skim the more "math-y" parts, but you should understand the general aim of the equations. 
- [Selection Mechanisms and Their Consequences: Understanding and Addressing Selection Bias](https://www.researchgate.net/publication/343541124_Selection_Mechanisms_and_Their_Consequences_Understanding_and_Addressing_Selection_Bias): **Optional** overview of selection bias in causal inference, for those curious. 

**Video:**
- Watch this quick [video](https://www.youtube.com/watch?v=MvS_wYtT7Yw). 

**Notebook:** 
- Walk through and complete the notebook `proj2/week2_selection_bias.ipynb` to become familiar with how to evaluate and correct for selection bias in our dataset. Note: We will not be requiring you to submit this notebook, but strongly encourage you to go through it. 

*[For Assignment 2](https://web.stanford.edu/class/cs197/assignments/project.html#related-work) (Due April 16):* You will explore related work in this field. Please see the "nearest-neighbor" papers for this project [here](https://docs.google.com/document/d/10Qe-m0KK5pyykERt7R2zzxDdnpgx3WI7D3OtGlFqDv4/edit?usp=sharing). 

*For Progress Report 2 (Due April 19):* Meet with your project group and submit what you all want to accomplish for Week 4. [See the website](https://web.stanford.edu/class/cs197/assignments/project.html#progress-reports) for how we expect project reports to be structured.



### Week 4 (April 20 - 26): Does standard Group DRO improve cross-hospital generalization?

(Note: The `week3_missingness.ipynb` notebook is now entirely optional / not required.)

**Goal:**  
How does model performance degrade under cross-hospital distribution shift, and does standard Group DRO help? 

**Readings:**
- [Distributionally Robust Neural Networks for Group Shifts](https://arxiv.org/abs/1911.08731) - read in-depth  
- [The Impact of Multi-Institution Datasets on the Generalizability of Machine Learning Prediction Models in the ICU](https://pmc.ncbi.nlm.nih.gov/articles/PMC11469625/pdf/ccm-52-1710.pdf) - read in-depth  
- [A comparison of approaches to improve worst-case predictive model performance over patient subpopulations](https://www.nature.com/articles/s41598-022-07167-7) - skim
- [Benchmarking Distribution Shift in Tabular Data
with TableShift](https://proceedings.neurips.cc/paper_files/paper/2023/file/a76a757ed479a1e6a5f8134bea492f83-Paper-Datasets_and_Benchmarks.pdf) - skim 
- [Distributionally Robust Optimization with Probabilistic Group](https://www.semanticscholar.org/paper/Distributionally-Robust-Optimization-with-Group-Ghosal-Li/14de7ffbbc1ce1afd75f32e43787b92a0165a5a7) - skim


**Tasks:**  

1. **Diagnose selection bias / distribution shift across hospitals.**  
   Investigate how patient populations and data collection differ across hospitals. Compare distributions of:
   - demographics (e.g., age, sex, ethnicity)  
   - outcome prevalence (e.g., mortality rate)  
   - lab or vitals values 
   - hospital characteristics
   - missingness summaries (e.g., fraction of missing features)  

   Use statistical tests (e.g., KS tests) and visualizations (seaborn plots) to assess whether hospitals differ meaningfully.  
   Briefly interpret your findings: what kinds of **selection mechanisms** (e.g., differences in patient populations, measurement practices, or healthcare access) might explain these differences?


2. **Measure baseline generalization (ERM).**  
   Train a mortality prediction model using standard ERM (empirical risk minimization) on a subset of hospitals and evaluate it on held-out hospitals.

   Compare:
   - internal validation performance  
   - external (held-out hospital) performance  
   - worst-hospital performance  

   Use metrics that are common in existing works such as AUROC, AUPRC, and optionally fairness metrics.  
   Plot results clearly using seaborn.

   Repeat this process **at least 5 times** (with different hospital train-heldout splits) to assess variability and generalization trends.


3. **Evaluate standard Group DRO.**  
   Train Group DRO models using at least two different group definitions, for example:
   - demographic groups (e.g., sex, ethnicity, or sex × ethnicity)  
   - hospital ids, or hospital region  


4. **Compare Group DRO to ERM.**  
   Compare ERM and Group DRO in terms of:
   - worst-hospital performance  
   - average external hospital performance  

   Plot results clearly using seaborn.  
   Does Group DRO improve robustness to cross-hospital distribution shift? How does this comapre with the findings of the related works above? 


5. **Reflection.**  
   Based on your results, answer:
   - Does standard Group DRO improve generalization across hospitals?  
   - Are the largest failures captured by the groups you defined?  
   - Do you see any particular areas in which Group DRO could be improved upon? 

**Deliverables:**
* Progress Report 3 (Due April 26) - a minimum 2-page writeup plus a notebook of all the completed tasks above. As a reminder: $\checkmark+$ = 100% indicates you went above and beyond; $\checkmark$ = 95% indicates basic completeness. 
* Introduction (Due April 23) - see website.

### Week 5 (April 27 - May 3): Constructing selection-aware groups

**Goal:**  
Can we improve Group DRO performance by constructing better (potentially latent) groups using proxy selection variables? 


Recall that standard Group DRO optimizes:
    $$\min_f \max_{g \in G} R_g(f)$$
    $$\text{where } R_g(f) = E[ l(f(X), Y) | G = g ]$$

Here, $g$ is a predefined group label (e.g., hospital ID).

However, hospital identity may not fully capture the dataset shift, as distribution shift across hospitals often arises from complex selection mechanisms (e.g., patient mix, measurement practices). Motivated by this, we instead construct alternative groupings $G$ by identifying variables $C_1 \ldots C_k$ that could be causing dataset shift. We will refer to these variables as **proxy selection variables**. 


**Tasks:**

1. **Identify proxy variables for dataset shift**  Select a training dataset consisting of N hospitals and define a held-out hospital (or set of hospitals) for evaluation. We want to identify a set $K=3-10$ proxy selection variables $C_1, ..., C_K$ that may capture selection bias / dataset shift. To do so, you may chose to 
learn a classifier to find which variables differ most between the training dataset $Y=1$ and held-out dataset $Y=0$ (Note: Obviously you won't have the held-out hospital data in a realistic setting, but this will help us build intuition for what a reasonable proxy selection varaiable set would be!) You could also identify the proxy variables by computing feature-wise distribution shifts between train and heldout data (e.g., using KS tests, mean differences). 

   
   Briefly analyze your finding. Do you think these variables  reflect selection bias / dataset shift that hospital ID wouldn't capture alone? 

2. **Construct alternative group labels - binning**  
   Use the identified proxy variables $C$ to define new groupings of patients. 

   - Discretize continuous variables (e.g., age bins, missingness levels)
   - Define groups as combinations of bins:$G =$ all observed combinations of $C$

   - Optionally include hospital ID: $G =$ all observed combinations of $C$ x  all observed IDs $H$

   Visualize these groups and describe them:
   - How many groups are there?
   - What is the size of each group?
   - Are there very small (rare) groups?



3. **Evaluate model performance across new groups: ERM**

   First, using your baseline ERM model from Week 4:
   - compute performance (AUC, etc.) within each group  
   - identify the worst-performing groups  
   Discuss: 
   - Which group reveals the worst-case failures? How does this compare to the worst case failure under just hospital ID groups alone? 
   - Do proxy groups identify failure modes not captured by hospital ID?

4. **Evaluate model performance across new groups: DRO**

   Train Group DRO using these groups. Compare to ERM and hospital-based Group DRO. If you are unable to train successfully because of group sizes, report that. Discuss your findings.

5. **Construct alternative group labels - clustering**  
   Instead of discretization, construct groups using clustering. This will establish a "latent" group label that would ideally capture dataset shift such that DRO will then learn a more robust model than predefined labels alone. 
   
    Use K-means, GMM, etc. to cluster training data into cluster labels for $G$ based on input data $C$ and hospital ID $H$. (Note: soft group labels - which we will explore next week) would just be the probability returned by these models!) Visualize / report cluster sizes, and investigate if the clusters correspond to patient populations or hospital practices. 

   Then, train Group DRO using these groups. Compare to ERM and hospital-based Group DRO. If you are unable to train successfully because of group sizes, report that. Discuss your findings.

6. **Reflection** Based on your results, discuss:
   - Do proxy-based groups capture model generalization failures better than hospital ID alone? What does say, if anything, about distribution shift? 
   - What does this suggest about how groups should be defined for robust learning?

---

**Deliverables:**
* Progress Report 4 (Due May 3) - a minimum 2-page writeup plus a notebook of all the completed tasks above 
* (Optional) Related Works - based on this project direction, refine your current related works section. You will need to do this eventually, so might be a good idea to work on that this week. 

### Week 6 (May 4 - 10): Soft group membership 

Last week, we explored latent group variables by clustering patients using proxy selection variables. However, those approaches assigned each patient to exactly one group or cluster. This week, we will test whether allowing patients to have soft group membership improves Group DRO performance. This is motivated by the PG-DRO paper, which argues that hard group labels may lose information when group membership is ambiguous. Instead of forcing each example into one group, soft Group DRO lets each example contribute partially to multiple groups.

**Goal:** Does soft group membership improve cross-hospital generalization compared with hard Group DRO?

**Tasks:**
 1. **Soft Group DRO with hospital-label groups** First, implement a simple soft Group DRO baseline, similar to what was done in the PG-DRO paper. As a baseline, learn soft group membership $P(h | x)$ using a classifier predicting $h$ from covariates $x$ as the $Q$ function from the paper. Also, define groups using hospital ID (the assumed domain variable) and outcome label (mortality) : $g = (H, Y)$ where H is hospital ID and Y is the outcome label. Report:
      - average held-out hospital performance
      - worst-hospital performance
      - worst-group performance
      - whether soft membership improves over hard hospital Group DRO

2. **Soft Group DRO with learned latent clusters** Next, use the clustering model from Week 5, let:

   $$q_i(g) = P(G_i = g | H_i, C_i)$$

   be the probability that patient i belongs to latent cluster g, as estimated by your clustering model. Here, $C_i$ represents all proxy selection variables for patient i based on the variables you identified in Week 5.

   Instead of hard Group DRO as last week, where each patient belongs to one group, train soft Group DRO:

    $$\min_f  \max_{g \in G}
        [ \frac{\sum_{i=1}^n q_{i}(g)\cdot l(f(X_i), Y_i)}{\sum_{i=1}^n q_{i}(g)} ]$$
   Again, report:
   - average held-out hospital performance
   - worst-hospital performance
   - worst-group performance
   - whether soft membership improves over hard hospital Group DRO

3. **Add group-size adjustment** Small groups may have noisier risk estimates. Following the Group DRO / PG-DRO literature, try adding a group-size adjustment term $ L / \sqrt n_g$ to both the models trained in 1. and 2. Try several values of $L$. Report whether the adjustment improves worst-hospital or worst-group performance.

4. **Compare several soft group definitions** Note the term $q_{i}(g)$ is meant to capture distribution shift such that the resulting (soft) weighted DRO is robust to a new unseen g'. We hypothesized that ($H, C$) are the features driving distribution shift, i.e., $q_i(g) = P(G_i = g | H_i, C_i)$. We can compare to ther group definitions. Try multiple choices of soft groups. 
   1. Hospital-only soft groups: $q_i(g) = P(G_i = g | H_i)$

   2. Proxy-variable soft groups: $q_i(g) = P(G_i = g | C_i)$

   5. Error-aware soft groups: The error-aware version asks whether model failures reveal latent groups not captured by hospital ID or proxy variables alone. To capture this residual, let $l_i^{ERM}$ be the loss of the baseline ERM model $f_{ERM}$ on patient i : $l(f_{ERM}(X_i), Y_i)$. Then the group probability is $q_i(g) = P(G_i = g | H_i, C_i, l_i^{ERM})$. 

   For this task, you are free to use any unsupervised grouping method that learns some latent label $g$ like hierarchical clustering or unsupervised nearest neighbors. If you are using KNN, try playing around with the number of centroids. 

5. **Reflection** Now you can compare the following methods:
   1. ERM
   2. Hard Group DRO with hospital groups
   3. Hard Group DRO with Week 5 clusters
   4. Soft Group DRO with hospital/outcome groups
   5. Soft Group DRO with latent clusters
   6. Soft Group DRO with group-size adjustment
   7. Error-aware soft groups

   For each method, report:
   - average performance 
   - worst-hospital performance
   - worst-group performance

   Reflect: 

   - Does soft group membership improve over hard group membership?
   - Which group definition gives the best worst-hospital performance?
   - Does adding group-size adjustment help, or does it over-penalize small groups?
   - Is hospital ID plus C sufficient to explain dataset shift?
   - If not, what residual features seem to capture additional latent shift?

**Deliverables:**
* Progress Report 5 (Due May 10) - a minimum 2-page writeup plus a notebook of all the completed tasks above 


### Week 7 (May 11 - 17): Final Method Experiments
This is your final week of experiments before we begin writing up the project. The goal this week is to continue testing our project's hypothesis: Can proxy variables such as demographics, severity, missingness, hospital characteristics, or measurement patterns, better capture the sources of hospital shift than hospital ID alone, and thus improve worst-case cross-hospital generalization beyond our current baselines (i.e., ERM)?

So far, we have compared ERM and Group DRO using hospital ID as the group variable. We also tried using proxy-variable-based groups, but the results were not clearly better than ERM. This does not necessarily mean using proxy variables is a bad idea. When a method does not work, an important part of research is figuring out *why* it did not work. Before we conclude this project, we should test several possible explanations.


This week, we will investigate four possible explanations for why proxy-based Group DRO may not have improved performance.

1. Revisit and justify the choice of proxy variables U.
I am not yet convinced that the current proxy variables you all selected (age, height, and weight), are the variables most responsible for cross-hospital shift. For example, there should be a bar plot or table that quantitatively ranks all possible proxy variables by how much they explain hospital differences. From the top candidates, you can reason qualitatively which top few variables are likely to cause distribution shift - for example, age makes sense, but why height and weight? This can be done by a classifier predicting hospital ID or plotting all statistical tests ranked and selecting the top variables. If you find a better set of proxy variables, you will need to rerun the proxy-based Group DRO experiments using the improved U. [Note: Proxy variables are only used during training, so we don't care about which" variables are "feasible" to obseve in unseen hospitals!]

2. Before concluding that proxy-based groups do not work, we need to make sure our Group DRO implementation itself is reasonably stable. Use hospital ID as the group variable first, try stabilizing the Group DRO model using early stopping and stronger $\ell_2$ regularization (look at Table 1 in the [original paper](https://arxiv.org/pdf/1911.08731), which showed that regularization is helpful in practice to reduce gDRO's instability).


3. Group DRO can become unstable when it is optimizing over many small groups, because the worst-group loss may be very noisy. For every grouping strategy we've tried so far (i.e., hospital ID, proxy-based) report the number of groups, minimum group size, and average number of samples for the minimum group for your model (i.e., if you model is training in batches, this is minimum group size divided by the batch size; if your model is running K-fold cross validation, this is minimum group size divided by K, etc. ) If sparsity seems like a problem, you can try one or more of the following:
   * merge small groups into an "Other: category (I think you have already tried this)
   * alter the clustering method for better evenness (I do not know how many clusters K you have for proxy-based clusters right now)
   * something else? 

4. If proxy-based Group DRO still does not work, we should try methods that use $U$ in other ways way. For the first idea, let $C$ be the proxy variables we expect to observe in the testing (held out hospital) distribution - for example, age, or the number of beds. We will try a simple density reweighting method using $C$ where we upweight training patients who look similar to the held-out target hospitals. 
   * First, train a model using to predict whether a patient comes from the held-out hospital set: $p(S=1 \mid C_i)$ where $S_i = 1$ means patient i is from the target hospitals.
   * Then define weights: $w_i = \frac{p(S=1 \mid C_i)}{1-p(S=1 \mid C_i)}$. We will then learn the **propensity weighted ERM**: $$\min_f E[w_i \cdot \ell(f(X_i),Y_i)]$$(Side note: if $C$ causes distribution shift, then the weighted ERM actually is equal to if we had optimized ERM under the held out target distribution). 
   * You will probably need to clip large weights to prevent training instability (clip $w_i$ such that it can't be too small or too large). 
   * If this helps, then $C$ may capture train-test hospital shift.
   * You can also try on the full $U$ if you want, since we actually observe all variables in the held-out distribution, to test your theory on $U$ being the cause of distribution shift. 
5. We will try group DRO but based on groups defined by a single variable if the patient looks very hospital-specific.
   * First, train a multi-label classifier model using all proxy variables to predict hospital ID: $r_i = \max_h p(H=h \mid U_i)$. Note, high $r_i$ means the patient looks strongly associated with one hospital. 
   * We will treat the variable $r_i$ as the group, i.e., binning $r_i$ into low, medium, and high groups across all patients. Then traing group DRO on these 3 bins as $g$. 
   * Using $r_i$, we can also try weighted ERM: $$\min_f E[ r_i \ell(f(X_i),Y_i)]$$. 


### Week 8 (May 18–24): Collecting Results, Baselines, and Tables

This week, the goal is to finish collecting the main experimental results and turn them into clear, conference-style tables and figures. At this point, we should be moving from "trying ideas" to organizing evidence. A strong paper needs trustworthy baselines, clean comparisons, and ablations that help the reader understand what part of the method is actually helping.

Please run and organize the final set of baseline methods and proxy-variable methods. At minimum, the main results table should compare ERM, hospital-ID Group DROimproved/regularized hospital-ID Group DRO, proxy Group DRO, soft-label group DRO, propensity-weighted ERM, and p(h|U) weighted ERM. You should also include one label-free baseline, such as [CVaR/superquantile DRO](https://arxiv.org/abs/2107.09044) or[ Just Train Twice (JTT)](https://arxiv.org/abs/2010.05893), both which simply upweights the highest-loss training examples without using group labels. You may use existing [libraries](https://github.com/namkoong-lab/dro) for this.

By the end of the week, please prepare **polished** tables and plots. The main table should report average held-out hospital performance, worst-held-out-hospital performance, and standard deviations across random seeds or hospital splits.

This week you should produce a rough outline of the Results section, as it is often easier to write the paper once we know what the final results say. Start by deciding what the main "story: of the results is: Did proxy variables help? Did they help only in some settings? 



### Week 9 (May 25–31): Writing the First Full Draft

This week, the goal is to turn the results into a first full draft of the paper. Focus first on the Results and Methods sections, since these should be grounded directly in the experiments you have already run. The Methods section should clearly explain the task setup, hospital split strategy, prediction target, baselines, proxy-variable construction, and each robustness method we compare. The Results section should walk the reader through the main table, the worst-hospital analysis, and the ablations.

After the Results and Methods are drafted, work on the Introduction and framing. The paper should motivate the problem as follows: models trained on one set of hospitals often fail to generalize to new hospitals; hospital ID is a common but crude way to define robustness groups; proxy variables \(U\), such as demographics, severity, missingness, measurement patterns, or hospital characteristics, may better capture the mechanisms driving hospital shift. Our project asks whether incorporating these proxy variables into training can improve worst-case out-of-hospital generalization.

By the end of this week, please have a complete rough draft. It does not need to be perfectly polished, but it should include all major sections: Introduction, Related Work if applicable, Methods, Experimental Setup, Results, Discussion, and Limitations. The most important thing is that the draft has a clear narrative and that every major claim is supported by a table, plot, or ablation.



### Week 10 (June 1–7): Final Writing, Polishing, and Presentation

This week, the goal is to finish the paper and prepare the final presentation. 