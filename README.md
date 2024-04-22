# The Brewery Project

[The Brewery Project - Official Website](https://the-brewery-project.github.io/The-Brewery-Project/)

## General Procedures for Working with the Website

0. The following assumes a *forked* or *cloned* repository.
1. Install Quarto and `install.packages("rmarkdown")` if you haven't already (**VSCode** is an additional option to RStudio/Quarto).
2. Whenever you do git pull, always rebase: `git pull --rebase`.
3. If attempting something *major*, create a branch for any work that could have a major effect (aka the `.qmd` files), and start your branch name with initials, e.g. `CK_Branch_1`.
4. **ALWAYS** keep your git updated, do `git pull` always before any committing, pushing, branches, etc.
5. Recommended to perform `quarto render` on the CLI/bash before pushing.
6. When pushing, a force push may be necessary. Make sure to pay attention to which branch is being pushed.
7. Github will automatically turn the `docs/` folder file changes into the website.

## Example Command Line and Work Procedure

1. cd into repository
2. `git pull --rebase`
3. pay attention to branch; will most likely be working on branch `main`
4. make contributions
5. it's good practice to make `git status` checks
6. optional, but recommended, apply `quarto render`
7. `git add [file(s)]` (can use `git add .` to add all available files)
8. commit contribution: if text editor set to appear to receive and edit message, use `git commit`
9. commit contribution: (or alternatively), use `git commit -m "message"`
10. finally, `git push origin [branch]`; will most likely want to use `git push origin main`

## Data Sources

### [Open Brewery DB](https://www.openbrewerydb.org/)

- [GitHub Directory](https://github.com/openbrewerydb/openbrewerydb/)
  - [csv files](https://github.com/openbrewerydb/openbrewerydb/tree/master/data)
- [API Documentation](https://www.openbrewerydb.org/documentation)
- See `open-brewery-db-exploration.ipynb` for exploratory measures in extracting data via API
- See `open-brewery-db-extractor.py` for final extraction from the API into `open-brewery-db.csv`
  
### [Top College Towns](https://listwithclever.com/research/best-college-towns-2021/)

- See `top-colleges-exploration.ipynb` for exploratory measures in extracting data via API
- See `top-colleges-extractor.py` for final extraction from the API into `top_colleges.csv`

### [Top Metropolitan Cities](https://worldpopulationreview.com/us-cities)

- See `The_Brewery_Project_EDA_(Metro).ipynb` for exploratory measures in extracting data via API
- See `the_brewery_project_web_scraping_(metro).py` for final extraction from the API into `metropolianCities.csv`

### [Top Tech Hubs](https://www.zdnet.com/education/computers-tech/top-tech-hubs-in-the-us/)

- See `The_Brewery_Project_EDA_(Tech).ipynb` for exploratory measures in extracting data via API
- See `the_brewery_project_web_scraping_(tech).py` for final extraction from the API into `techHubs.csv`

### [Ski Resorts](https://en.wikipedia.org/wiki/List_of_ski_areas_and_resorts_in_the_United_States)

- See `The_Brewery_Project_EDA_(Ski_Resorts).ipynb` for exploratory measures in extracting data via API
- See `the_brewery_project_web_scraping_(ski_resorts).py` for final extraction from the API into `ski_resorts.csv`

### [National Parks](https://www.nationalparktrips.com/parks/us-national-parks-by-state-list/)

- See `The_Brewery_Project_EDA_(National_Parks).ipynb` for exploratory measures in extracting data via API
- See `the_brewery_project_web_scraping_(national_parks).py` for final extraction from the API into `national_parks.csv`
  
### [Census](https://www.census.gov/data/tables/time-series/demo/popest/2020s-counties-detail.html)

- See `the_brewery_project_web_scraping_(national_parks).py` for final extraction from the API into `censusData.csv`

## Directory

- **data/**: where all datasets there were used and created reside
- **docs/**: Where the GitHub site is being hosted from, and what gets generated when we apply `Quarto Render` into a webpage HTML. DO NOT manually change or touch
- **exploratory/**: where the exploratory `.ipynb` (python notebooks) reside
- **images/**: where the images reside (contains subfolders for particular pages)
- **model/**: where the modeling `.pkl` reside
- **scripts/**: where the `.py` data and model scripts reside
- **.gitignore**: specifies which files shouldn't be pushed to GitHub
- **README.md**: file which produced this REAMDE
- **The-Brewery-Project.Rproj**: file which controls the project within Rstudio
- **_quarto.yml**: file which holds the settings for the quarto website
- **about.qmd**: generates into the *about* page of the website
- **conclusion.qmd**: generates into the *conclusion* page of the website
- **data_exploration.qmd**: generates into the *data exploration* page of the website
- **diabetologia.csl**: file which controls the citation settings within the generated webpages
- **index.qmd**: necessary file within for generating the website
- **introduction.qmd**: generates into the *introduction* page of the website
- **models_implemented**: generates into the *models implemented* page of the website
- **references.bib**: file which holds the bibliography
- **styles.css**: file which controls the styling for the website

### Datasets (data/)

<div class="cell-output-display">
<table class="table table-sm table-striped small">
<colgroup>
<col style="width: 20%">
<col style="width: 70%">
</colgroup>
<thead>
<tr class="header">
<th style="text-align: left;">filename</th>
<th style="text-align: left;">purpose</th>
</tr>
<tr class="odd">
<th style="text-align: left;">best_hyper_results.csv</th>
<th style="text-align: left;">contains the model metrics of the hypertuned models from the full model hotspot testing</th>
</tr>
<tr class="even">
<th style="text-align: left;">brewery_count_by_state.csv</th>
<th style="text-align: left;">contains the total breweries per state</th>
</tr>
<tr class="odd">
<th style="text-align: left;">censusData.csv</th>
<th style="text-align: left;">contains the full census API extraction with some additional information</th>
</tr>
<tr class="even">
<th style="text-align: left;">city_level.csv</th>
<th style="text-align: left;">contains the consolidated data from all datasets with additional metrics, almost model ready</th>
</tr>
<tr class="odd">
<th style="text-align: left;">coll-town-count-by-state.csv</th>
<th style="text-align: left;">contains the total college towns per state</th>
</tr>
<tr class="even">
<th style="text-align: left;">hotspot_hyper_logistic_results.csv</th>
<th style="text-align: left;">contains the hypertuned best logistic regression model metrics for full model</th>
</tr>
<tr class="odd">
<th style="text-align: left;">hotspot_hyper_tree_results.csv</th>
<th style="text-align: left;">contains the hypertuned best decision tree classification model metrics for full model</th>
</tr>
<tr class="even">
<th style="text-align: left;">hotspot_round_1.csv</th>
<th style="text-align: left;">contains the model metrics of the default models with non-scaled data from the full model hotspot testing</th>
</tr>
<tr class="odd">
<th style="text-align: left;">hotspot_round_2.csv</th>
<th style="text-align: left;">contains the model metrics of the default models with scaled data from the full model hotspot testing</th>
</tr>
<tr class="even">
<th style="text-align: left;">metropolianCities.csv</th>
<th style="text-align: left;">contains the scraped data for the top metropolitan areas (cities)</th>
</tr>
<tr class="odd">
<th style="text-align: left;">model_data.csv</th>
<th style="text-align: left;">contains the model ready data derived from city_level.csv</th>
</tr>
<tr class="even">
<th style="text-align: left;">national_parks.csv</th>
<th style="text-align: left;">contains the scraped data for national parks in the US</th>
</tr>
<tr class="odd">
<th style="text-align: left;">open-brewery-db.csv</th>
<th style="text-align: left;">contains the full US API extraction from open brewery db</th>
</tr>
<tr class="even">
<th style="text-align: left;">ski_resorts.csv</th>
<th style="text-align: left;">contains the scraped data for ski resorts in the US</th>
</tr>
<tr class="odd">
<th style="text-align: left;">techHubs.csv</th>
<th style="text-align: left;">contains the scraped data for ski resorts in the US (cities)</th>
</tr>
<tr class="even">
<th style="text-align: left;">top_colleges.csv</th>
<th style="text-align: left;">contains the scraped data for the top colleges towns in the US (cities)</th>
</tr>
</tbody>
</table>
</div>

### Exploratory Notebooks (exploratory/)
<div class="cell-output-display">
<table class="table table-sm table-striped small">
<colgroup>
<col style="width: 20%">
<col style="width: 70%">
</colgroup>
<thead>
<tr class="header">
<th style="text-align: left;">filename</th>
<th style="text-align: left;">purpose</th>
</tr>
<tr class="odd">
<th style="text-align: left;">open-brewery-db-exploration.ipynb</th>
<th style="text-align: left;">an exploratory analysis from open brewery db API data (US)</th>
</tr>
<tr class="even">
<th style="text-align: left;">The_Brewery_Project_EDA_(Metro).ipynb</th>
<th style="text-align: left;">an exploratory analysis of the metroplitan data</th>
</tr>
<tr class="odd">
<th style="text-align: left;">The_Brewery_Project_EDA_(National_Parks).ipynb</th>
<th style="text-align: left;">an exploratory analysis of the national parks data</th>
</tr>
<tr class="even">
<th style="text-align: left;">The_Brewery_Project_EDA_(Ski_Resorts).ipynb</th>
<th style="text-align: left;">an exploratory analysis of the ski resorts data</th>
</tr>
<tr class="odd">
<th style="text-align: left;">The_Brewery_Project_EDA_(Tech).ipynb</th>
<th style="text-align: left;">an exploratory analysis of the tech hubs data</th>
</tr>
<tr class="even">
<th style="text-align: left;">top-colleges-exploration.ipynb</th>
<th style="text-align: left;">an exploratory analysis of the college towns data</th>
</tr>
</tbody>
</table>
</div>

### Python Scripts (scripts/)
<div class="cell-output-display">
<table class="table table-sm table-striped small">
<colgroup>
<col style="width: 20%">
<col style="width: 70%">
</colgroup>
<thead>
<tr class="header">
<th style="text-align: left;">filename</th>
<th style="text-align: left;">purpose</th>
</tr>
<tr class="odd">
<th style="text-align: left;">brewery-hotspots.py</th>
<th style="text-align: left;">cleans, prepares, and exports open brewery db data for use in dataset_manager.py</th>
</tr>
<tr class="even">
<th style="text-align: left;">dataset_manager.py</th>
<th style="text-align: left;">imports and consolidates all datasets needed for modeling into an almost model ready format</th>
</tr>
<tr class="odd">
<th style="text-align: left;">pdf_manager.py</th>
<th style="text-align: left;">manages the pdfs downloadable from the project website</th>
</tr>
<tr class="even">
<th style="text-align: left;">the_brewery_project_web_scraping_(census).py</th>
<th style="text-align: left;">cleans, prepares, and exports the census data for use in dataset_manager.py</th>
</tr>
<tr class="odd">
<th style="text-align: left;">the_brewery_project_web_scraping_(metro).py</th>
<th style="text-align: left;">cleans, prepares, and exports the metropolitan data for use in dataset_manager.py</th>
</tr>
<tr class="even">
<th style="text-align: left;">the_brewery_project_web_scraping_(national_parks).py</th>
<th style="text-align: left;">cleans, prepares, and exports the national park data for use in dataset_manager.py</th>
</tr>
<tr class="odd">
<th style="text-align: left;">the_brewery_project_web_scraping_(ski_resorts).py</th>
<th style="text-align: left;">cleans, prepares, and exports the ski resort data for use in dataset_manager.py</th>
</tr>
<tr class="even">
<th style="text-align: left;">the_brewery_project_web_scraping_(tech).py</th>
<th style="text-align: left;">cleans, prepares, and exports the tech hub data for use in dataset_manager.py</th>
</tr>
<tr class="odd">
<th style="text-align: left;">top-colleges-extractor.py</th>
<th style="text-align: left;">cleans, prepares, and exports the college town data for use in dataset_manager.py</th>
</tr>
</tbody>
</table>
</div>

## Compiling

### Website and Report
1. `cd` into project root directory (will likely look like `The-Brewery-Project (main)`)
2. run `quarto render`
3. `cd` into `scripts`
4. run `python pdf_manager.py`
5. `cd` back into project root directory (`cd ..`)

### Data Management
1. `cd` into project root directory (will likely look like `The-Brewery-Project (main)`)
2. `cd` into `scripts`
3. run `python open-brewery-db-extractor.py`
4. run `python dataset_manager.py` (for merging together all `csv` files together into a dataset *almost* fit for modeling)
  
## Project Management

- [**Google Colab**](https://colab.research.google.com/drive/1ibfuagpEE6WWkICI7EIa3ieKFlFbBQqU?authuser=1)
- [**Google Document**](https://docs.google.com/document/d/1eNdriCLmfu4RV5lMJWjYJYKHs_v17lRWWTFNnlEVmzs/edit)
