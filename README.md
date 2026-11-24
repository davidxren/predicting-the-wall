# Predicting the Wall: Early Pacing and Marathon Slowdown

David Ren, CS506, Fall 2026

## Status Update

For October, I finished collecting and cleaning the data, building the features, and making the first two figures.

For November, I trained logistic regression and gradient boosting models and compared different feature sets, tree sizes, and slowdown cutoffs. I also added two figures showing model performance.

All six tests pass locally. The models have been evaluated on 2023, with 2024 reserved for final testing.

## What the project is about

Two runners can hit the halfway mark of a marathon at the same time having run very different first halves. One of them might have run every 5K at the same pace, while the other went out hard and got a little slower at each checkpoint. I want to know whether that difference tells you anything about who is going to fall apart in the second half, once you already know the halfway time.

I run, so I have a personal interest in this. The usual way people estimate a finish is to take the halfway time and double it, and I want to see whether the shape of the first half adds anything to that. The result will be a Python analysis anyone can rerun, with the figures and the model comparison in this README.

## What I'm predicting

The thing I'm trying to predict is how much slower a runner's second half was than their first half, as a percentage:

    first_half_seconds = HALF
    second_half_seconds = Finish_Net - HALF
    slowdown_percent = 100 * (second_half_seconds / first_half_seconds - 1)

So a runner who did the first half in 90 minutes and the second half in 99 minutes slowed down 10%. In the proposal I set the cutoff for a substantial slowdown at 10%, with 5% and 15% as checks. After looking at the data (figure 1 below), 10% turns out to be close to the median in most years, meaning about half the field would count as having hit the wall, which isn't what I meant by substantial. So I'm using 15% as the working cutoff for the figures and probably for the models. All three labels are computed and I'll make the final call once I see how the models behave at each one. Either way, the cutoff only measures how much someone slowed down. It doesn't say anything about why.

## Where the data comes from

I'm using the Boston Marathon dataset that Carnegie Mellon's Department of Statistics & Data Science publishes for teaching: [Boston Marathon times, 2021-2024](https://cmustatistics.github.io/data-repository/sports/boston-marathon.html). Each row is one runner in one year, with their age, an M/F field, their cumulative time at every 5K checkpoint from 5K to 40K, their halfway time, their net finish time, and the year. All the times are in seconds.

The data was originally scraped from the Boston Athletic Association's results pages by Brandon Onyejekwe and Eric Gerber at Northeastern for a [paper on predicting finish times](https://github.com/bonyejekwe/Marathon_Predictor). I'm using their data, and all the code in this repo is mine. My question is also different from theirs.

`src/download.py` downloads the file, saves it to `data/raw/` without changing anything, and writes `data/manifest.json` with the URL, the download time, the file size, and the sha256 hash. The raw file is gitignored, but the manifest is committed so anyone can check they have the same file I used. The script also stops with an error if any of the 14 columns from CMU's data dictionary is missing.

Here is what's in the raw file:

| Year | Race date | Rows |
| --- | --- | --- |
| 2021 | Oct 11 | 15,121 |
| 2022 | Apr 18 | 24,489 |
| 2023 | Apr 17 | 26,028 |
| 2024 | Apr 15 | 25,262 |
| Total | | 90,900 |

The 2021 race was held in October because of COVID and had a smaller field, which is why that year has fewer rows. Before writing any cleaning rules I ran `src/inspect_raw.py`, which prints the column types, missing values per column per year, the range of every time column, the values in M/F, the age range, and the number of duplicate rows. The file turned out to be very clean. There are no missing values in any column, every time column is already an integer, nothing is zero or negative, and there are no exact duplicates. M/F has three values: M (51,208), F (39,622), and X (70). Ages run from 18 to 82.

## Cleaning

Even though the file looked clean, I wrote the rules anyway so the cleaning is explicit and repeatable. `src/validate.py` applies four rules in order and counts how many rows each one drops, per year. A row only gets counted against the first rule that catches it, so the counts add up to the total.

| Rule | Why |
| --- | --- |
| Missing 5K, 10K, 15K, 20K, HALF, or Finish Net | These are the columns the features and the label are built from. 25K through 40K are allowed to be missing since they're only used for plots. |
| A zero or negative time | Impossible for a real split. |
| A cumulative time lower than the one before it | You can't reach 40K after you've finished. This is a timing mat problem. |
| An exact duplicate row | The same performance counted twice. |

In total 2 rows were dropped, both from 2023, both because the finish time was lower than the 40K time. That leaves 90,898 rows. The log is written to `data/processed/exclusions.csv`.

The Name column is dropped at this step so that nothing downstream can use it. I'm also deliberately keeping slower runners. Even seven-ish hour finishes are real results, and dropping outliers by eye would push the dataset toward faster runners, who are the people least likely to hit the wall.

## Features

`src/features.py` turns the clean rows into the inputs for the models and the label. Everything used as an input is known by the halfway point. The finish time and the 25K through 40K splits are only used to compute the label and for the retrospective plots.

| Feature | How it's calculated |
| --- | --- |
| Segment paces (4) | Seconds per km for 0-5K, 5-10K, 10-15K, and 15-20K, e.g. `(time_10K - time_5K) / 5` |
| Halfway pace | HALF / 21.0975 |
| Pace variation | Standard deviation of the four segment paces divided by their mean |
| Slowing trend | Percent change from the 0-5K pace to the 15-20K pace |
| Age, M/F | Tested separately as additional inputs |
| Label | `slowed_5`, `slowed_10`, `slowed_15`: 1 if the slowdown is at least that percent |

Here is the share of runners over each cutoff, by year, along with the start temperature from [findmymarathon.com](https://www.findmymarathon.com/weather-detail.php?zname=boston+marathon):

| Year | Median slowdown | Over 5% | Over 10% | Over 15% | Start temp |
| --- | --- | --- | --- | --- | --- |
| 2021 | 11.7% | 78% | 57% | 39% | 60 F, 88% humidity |
| 2022 | 8.3% | 67% | 42% | 26% | 46 F |
| 2023 | 5.8% | 55% | 30% | 16% | 48 F |
| 2024 | 13.6% | 84% | 64% | 45% | 56 F |

The base rate moves around a lot from year to year, and the two warmer races have the highest shares. With only four race days there's no way to use temperature as a real feature, so it stays in the discussion. It does matter for the evaluation though, because 2024 is my planned test year and its base rate is quite different from the 2021 and 2022 training years. That's why I'm planning to report calibration along with average precision.

## Figures

![Slowdown by year](figures/slowdown_by_year.png)

Figure 1 shows the distribution of slowdown percent for each year with the three cutoffs marked. The y axis is the share of that year's runners, so the four years are comparable even though 2021 has a smaller field. The whole distribution shifts from year to year. In 2023 the peak is around 4% and most people stay under 10%, while in 2024 the center is past 10% with a long tail to the right. The 10% line goes through the middle of the field in three of the four years, which is the reason I moved to 15%.

A few things to be aware of with this figure. The x axis is clipped at -20% and 80%, and the 0.2% of runners outside that range are drawn at the edge. Only finishers are in the data, so anyone who dropped out is missing. The bins are 2 points wide, and a different width would change the look without changing the shares.

![Pacing profiles](figures/pacing_profiles.png)

Figure 2 is the main one. For each 5K segment of the first half, it shows the median pace of runners who went on to slow down 15% or more (black) next to runners who didn't (gray), with the 25th to 75th percentile shaded. The panels are bands of halfway time, so within each panel you're comparing runners who reached halfway at roughly the same time. That's the whole point of the figure: if the two groups still look different once halfway time is held fixed, then the shape of the first half is adding information on top of the halfway time.

And they do look different. In every band, the runners who later slowed down ran the first 5K faster and were already running slower by 15 to 20K. The lines cross somewhere around 10 to 15K every time. The shaded bands overlap a lot though, so this is a tendency, and the models will be working with a noisy signal.

Things to keep in mind here: the y axis is different in each panel, so slopes can only be compared within a panel. All four years are pooled, which means the warm years contribute more of the slowed runners. The group sizes are lopsided in the fast bands (564 versus 2,467 in the 70 to 80 minute band). And this is a look back at runners who already slowed, so it isn't a prediction yet.

## Running the code

I'm using Python 3.12. To reproduce the results, set up a virtual environment and install the dependencies.

    python -m venv .venv

Activate the environment (Windows PowerShell):

    .\.venv\Scripts\Activate.ps1

On Mac or Linux:

    source .venv/bin/activate

Then run:

    python -m pip install -r requirements.txt
    python src/download.py
    python src/validate.py
    python src/features.py
    python src/plots.py
    python src/model.py
    python src/experiments.py
    python -m pytest -q

If you have make, run `make install`, `make all`, and `make test` instead.

The raw and processed data are gitignored. The figures are saved in `figures/`. GitHub Actions runs the tests on each push.


## Modeling

I started with logistic regression on halfway time alone, then added pace variation and slowing trend to see how much they helped. I also tried gradient boosting using those same three inputs. Age and M/F are left out for now since the main question is about pacing.

The models are trained on 2021 and 2022 (39,610 records) and validated on 2023 (26,026 records). I used 2023 to compare features and model settings. I'm saving 2024 for the final model evaluation.

I'm mainly looking at average precision (AP), with ROC AUC and Brier score as additional metrics. Only 15.64% of runners in 2023 slowed down by at least 15%, so a model that predicts no slowdown for everyone would already be right about 84% of the time.

## November results

All results below are from the 2023 validation set. The main cutoff is a 15% slowdown.

| Model | Average precision | ROC AUC | Brier score |
| --- | ---: | ---: | ---: |
| Constant baseline | 0.1564 | 0.5000 | 0.1548 |
| Logistic regression (halfway only) | 0.2812 | 0.6751 | 0.1467 |
| Logistic regression (with pacing) | 0.4534 | 0.7848 | 0.1153 |
| Gradient boosting | 0.4768 | 0.8046 | 0.1130 |

The constant baseline predicts the training slowdown rate for every runner. Adding the pacing features to logistic regression increased AP from 0.2812 to 0.4534. Gradient boosting did a little better, but most of the improvement came from adding the features rather than switching models.

### Feature comparison

| Logistic regression inputs | AP | ROC AUC |
| --- | ---: | ---: |
| Halfway only | 0.2812 | 0.6751 |
| Halfway + variation | 0.3997 | 0.7638 |
| Halfway + trend | 0.4464 | 0.7708 |
| Halfway + both | 0.4534 | 0.7848 |

Slowing trend was the more useful feature on its own, although using both gave the best result. Variation tells us how inconsistent someone's pace was, while trend tells us whether they were getting faster or slower.

### Tree size

| Maximum leaves | AP | ROC AUC |
| --- | ---: | ---: |
| 7 | 0.4768 | 0.8046 |
| 15 | 0.4746 | 0.8040 |
| 31 | 0.4723 | 0.8026 |

I kept the number of boosting iterations at 100 and the learning rate at 0.1. The larger trees didn't help on 2023, so I'm using 7 leaves for now. The differences are pretty small.

### Slowdown cutoff

| Cutoff | Positive rate | Halfway only (AP) | With pacing (AP) |
| --- | ---: | ---: | ---: |
| 5% | 55.0% | 0.6830 | 0.7986 |
| 10% | 29.5% | 0.4676 | 0.6261 |
| 15% | 15.6% | 0.2812 | 0.4534 |

The pacing features helped at all three cutoffs. AP depends on how common the positive class is, so the useful comparison here is between the two models at each cutoff, not between cutoffs.

### Performance figures

![Precision-recall curves](figures/precision_recall.png)

Figure 3 shows how precision and recall change as the prediction threshold changes. Both pacing models generally perform better than the halfway-only model. The dotted horizontal line is the 2023 slowdown rate.

![Calibration curves](figures/calibration.png)

Figure 4 shows whether the predicted probabilities match what actually happened. Each point represents a group of runners with similar predicted probabilities. Most points are below the diagonal, which means the models generally overestimate slowdown risk in 2023.

This may partly be because the training slowdown rate was 30.75%, compared to 15.64% in 2023. The models seem better at ranking runners by risk than estimating their exact probabilities.

## Timeline

| Period | Work |
| --- | --- |
| Oct 1 to 14 | Data collection and cleaning. Done. |
| Oct 15 to 28 | Features, initial figures, and October check-in. Done. |
| Oct 29 to Nov 25 | Modeling, experiments, performance figures, and November check-in. Modeling and figures done. |
| Nov 26 to Dec 8 | Evaluate on 2024, look at prediction errors, finish report and record presentation. |
| Dec 9 | Submit. |

## Limitations and next steps

The results look promising, but so far I've only compared models on 2023. The next step is to evaluate the selected models on 2024 without tuning them on that year's results.

Some runners may appear in multiple years. Splitting by year tests how well the model works on a later race, not necessarily on completely different runners.

The dataset only includes Boston Marathon finishers. It doesn't have information about training, fueling, injuries, or goal times, so I can't use the results to explain why someone slowed down. There are also only four race years, and the slowdown rates differ quite a bit between them.

For the final report I want to see whether the improvement from pacing holds up on 2024 and look at the runners the models get wrong.
