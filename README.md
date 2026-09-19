# CS197 Fall 2026
**Course Staff:** Kara Liu (contact: karaliu@stanford.edu)


Welcome to CS197, your onramp into computer science research!  

This course is based on a quarter-long project that you will complete in groups of 2-3. The course is divided into two phases:

* **Weeks 0-2: Onboarding.** 
   * Everyone must apply for access to the data and will get familiar with real-world data through an interactive notebook.
* **Weeks 3-10: Specialized Project Track.** 
   * In Weeks 3-4, each project will provide its own specialized curriculum through interactive notebooks and suggested readings. 
   * In Weeks 5-10, you will choose a specialized project to own for the rest of the quarter, culminating in a final presentation and report.

We provide four project options, all of which are focused on AI and use tabular healthcare data. The projects are summarized below, with the hyperlinks attached. We separate based on your AI experience level, but emphasize that they should not vary in workload required or the grading rubric applied.
* [Projects 1 & 2: Generalization](proj12_generalization/README.md)
   * *Project 1 // Beginner Level 👍 // Benchmarking Generalization Methods for Cross-Hospital Prediction* - This project will compare different existing strategies for improving model generalization across different hospital sites.
   * *Project 2 // Intermediate Level 🧩 // Feature Selection* - This project will create a new method for selecting predictive features for optimal model generalization across hospital sites.


* [Projects 3 & 4: Representation Learning](proj34_representation/README.md)
   * *Project 3 // Intermediate Level 🧩 //  Evaluating Numerical Preprocessing Methods for Tabular Representation Learning* - This project investigates efficacy of different strategies for embedding numerical values in a tabular representation model. 
   * *Project 4 // Advanced Level 🌀 // Information Retention in Tabular Representation Learning* - This project investigates what information is retained in a latent representation as we vary tabular data complexity. 

## Getting started 

### Data Access

In this project, we will be working with the [eICU Collaborative Research Database (eICU-CRD)](https://eicu-crd.mit.edu/), a large, multi-center critical care dataset containing de-identified patient data from intensive care units (ICUs) across the United States. 



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
    cd cs197_fall2026 # if you are not already there 
    pip install -r requirements.txt
    ```

3. Test that you can open the first notebook `week1_explore_eicu_data.ipynb` by launching (where if applicable, make sure you are in the correct conda environment): 
    ```
    jupyter notebook
    ```