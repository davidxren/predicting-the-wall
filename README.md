# Predicting the Wall: Early Pacing and Marathon Slowdown

David Ren | Solo project proposal

## Project description

Two runners can reach halfway in the same time after very different first halves. One might hold a steady pace, while the other starts faster and slows at each checkpoint. I want to test whether those differences help predict what happens in the second half.

This project will use historical Boston Marathon results to answer: Do first-half pacing patterns help predict substantial second-half slowdown beyond what halfway time alone tells us? As a runner, I am interested in whether the way someone reaches halfway matters as much as the time they reach it in.

I will build a Python analysis that collects race results, checks the split data, extracts pacing features, trains prediction models, and evaluates them on a later race year. The output will be a reproducible analysis with visualizations, not a live coaching app.

## Goals and prediction target

The primary goal is to test whether individual first-half segments improve average precision on held-out race results compared with using halfway time alone.

I will initially define substantial slowdown as taking at least 10% longer to run the second half than the first:

    first_half_seconds = HALF
    second_half_seconds = Finish_Net - HALF
    slowdown_percent = 100 * (second_half_seconds / first_half_seconds - 1)
    substantial_slowdown = slowdown_percent >= 10

For example, a runner completing the first half in 90 minutes and the second half in 99 minutes meets this definition. The 10% cutoff is a project definition, not a medical definition of "hitting the wall." It measures how much a runner slowed, not why. I will also check 5% and 15% cutoffs to see whether the findings depend heavily on this choice.

A secondary goal is to determine whether pace variation and an early slowing trend add useful information. I will compare models with and without those features. If time permits, I will repeat the prediction task at 10K and 15K to examine how performance changes as more of the race becomes observable.

## Data source and collection

I will use [Boston Marathon times, 2021–2024](https://cmustatistics.github.io/data-repository/sports/boston-marathon.html), published by Carnegie Mellon's Department of Statistics & Data Science. Its documentation identifies one row as one runner's performance in a particular year, with age, a recorded M/F field, cumulative checkpoint times from 5K through 40K, halfway time, net finish time, and race year. The times are documented in seconds. These are checkpoints within the same marathon, not the runner's results from separate races.

The repository provides a compressed CSV download. I will write a Python script to download it, save an unchanged local copy, and load it into pandas. The script will check that the expected columns are present and record the source URL, download date, and file checksum. I will report the number of usable performances by year after validation rather than assume every published row is usable.

The data was originally collected from Boston Athletic Association results by Brandon Onyejekwe and Eric A. E. Gerber. Their [source project](https://github.com/bonyejekwe/Marathon_Predictor) provides the collection background and an alternate location for the Boston data. Their work studies finish-time prediction; my analysis will focus on slowdown classification and the value of early pacing detail. I will write my own processing and modeling code.

## Data cleaning and feature extraction

Although the published data is already tabular, I will validate it before modeling. Checks will cover missing required splits, nonnumeric values, nonpositive or decreasing cumulative times, and exact duplicate race records. I will document exclusions by year and will not remove a slow but valid performance simply because it looks unusual. Names will not be model inputs or appear in published plots.

Cumulative times will be converted into segment paces. For example, pace from 5K to 10K, in seconds per kilometer, is `(time_10K - time_5K) / 5`.

| Feature | Planned calculation |
| --- | --- |
| Halfway pace | Halfway time divided by 21.0975 km |
| Early segment paces | Pace for 0–5K, 5–10K, 10–15K, and 15–20K |
| Pace variation | Standard deviation of those four paces divided by their mean |
| Early slowing trend | Percentage change from 0–5K pace to 15–20K pace |
| Runner characteristics | Age and recorded M/F category, tested as additional inputs |

Every input will be available by halfway. Finish time and later splits will be reserved for outcome calculation and retrospective plots, not prediction inputs. Any learned preprocessing, such as scaling or demographic-value imputation, will be fitted on training data only.

## Modeling and evaluation

I will start with a logistic regression model using only halfway time. I will compare it with logistic regression using the pacing features and one tree-based model, such as gradient boosting. Age and the recorded M/F field will be added in a separate comparison so their contribution is not mistaken for the value of pacing detail. A constant prediction based on the training-set slowdown rate will provide a basic reference.

The planned split is 2021–2022 for training, 2023 for validation, and 2024 for final testing. Model settings and decision thresholds will be selected using the validation year. The final test year will remain untouched by model selection.

My main comparison will use average precision, which summarizes the precision-recall curve. I will report the slowdown rate alongside it, plus precision and recall at a threshold chosen on validation data. A calibration plot will compare predicted probabilities with observed slowdown frequencies. The goal is an honest comparison, including cases where the extra features do not help, rather than a promised accuracy before inspecting the data.

Returning runners may appear across years. I will describe this as a test on a later race year, not a test exclusively on previously unseen runners.

## Visualizations and final deliverables

The analysis will include checkpoint pacing profiles for runners with and without substantial slowdown, distributions of slowdown by year, and model-performance plots. Feature comparisons will show whether pacing detail improves on halfway time alone. An optional interactive plot will let a reader view an anonymized first-half pacing profile before revealing the runner's second half.

The final repository will contain collection, cleaning, feature extraction, training, evaluation, and plotting code. The README will become the final report, with environment details and instructions to run, test, and contribute to the code. I will include a Makefile for dependency setup and execution, tests for important calculations and validation rules, and a GitHub Actions workflow that runs those tests. The raw download will stay outside version control. I will also prepare the required 10-minute recorded presentation, upload it to YouTube, and link it near the top of the report.

## Project timeline

| Period | Work and expected output |
| --- | --- |
| Weeks 1–2: Oct. 1–14 | Implement collection; inspect columns and missing values; validate times; document usable records and the slowdown definition. |
| Weeks 3–4: Oct. 15–28 | Build pacing features, create initial visualizations, and fit the halfway-time baseline. Attend the October check-in with preliminary results. |
| Weeks 5–6: Oct. 29–Nov. 11 | Train the fuller models, compare feature sets on the validation year, and examine errors and sensitivity to the slowdown cutoff. |
| Weeks 7–8: Nov. 12–25 | Finalize model choices, evaluate on 2024, complete the main plots, and strengthen tests and reproducibility. Attend the November check-in. |
| Nov. 26–Dec. 8 | Finish the report, verify the project runs from a clean setup, and record the presentation. |
| Dec. 9 | Submit the final report and presentation. |

October: first check-in with preliminary visualizations, processing, and early modeling
November: second check-in with more finalized processing, model evaluation, and results

## Scope and limitations

The core project is one marathon, four race years, and one halfway prediction task. Additional checkpoints and the interactive plot are optional. If the scope needs to shrink, I will retain the baseline, full-feature logistic regression, year-based evaluation, and essential plots, and drop the extra model and extensions.

Because the label requires a finish time, the study will not predict who fails to finish. The listed data does not include training history, fueling, or individual race goals, so it cannot establish why someone slowed or whether a different strategy would have prevented it. Any conclusions will be about associations in these Boston results, not a universal pacing recommendation.
