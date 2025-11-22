
# Global Life Expectancy Analysis (1950–2023)
### An exploratory analysis of life expectancy evolution worldwide and in Italy

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2.0%2B-orange)](https://pandas.pydata.org/)
[![Matplotlib/Seaborn](https://img.shields.io/badge/Plotting-Matplotlib%20%26%20Seaborn-green)](https://matplotlib.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Data: Our World in Data](https://img.shields.io/badge/Data-Our%20World%20in%20Data-blueviolet)](https://ourworldindata.org/life-expectancy)

## Project Overview

This repository contains a full exploratory data analysis of **life expectancy at birth** from 1950 to 2023, based on the high-quality dataset from **Our World in Data** (sources: Human Mortality Database, UN World Population Prospects, Riley, Zijdeman et al.).

The project is divided into two main analyses:

1. **Global Analysis** – Worldwide progress, catch-up effect, health shocks (HIV/AIDS, COVID-19), convergence between countries  
2. **Italy Deep Dive** – Italy’s performance compared to Western Europe, G7, Americas, Mediterranean countries, BRICS+, and the world


## Key Findings ([Complete results here](results/readme.md))

### Global Trends
- World life expectancy rose from **46.4 years (1950)** to **73.2 years (2023)** → **+26.8 years**
- ![World & Continents Evolution](results/img/global_evolution.png)
- Biggest gain: **South Korea** (+62 years)
- ![Top 15 Biggest Gains](results/img/top15_increase.png)
- Sub-Saharan Africa is the only major region still lagging (HIV/AIDS setback 1990–2005 clearly visible)
- ![aa](results/img/hiv%20africa.png)
- The Great Chinese Famine (1959–61) and COVID-19 (2020-2021) caused the major global drop since 1950.
- ![aa](results/img/fall.png)
- 
### Italy
- From **63.7 years (1950)** to **83.7 years (2023)** → **+20 years**
- Consistently among the **top 7–8 countries worldwide** in 2023
- Outperforms G7 average by **+2.4 years**, Eastern Europe and BRICS+ by **10–12 years**
- ![Italy vs International Groups](results/img/italy_comparison_main.png)
- One of the most resilient Western countries during COVID-19


## Data Source

> Our World in Data – "Life expectancy" (Riley; Zijdeman et al.; Human Mortality Database; UN WPP)  
> Last updated: October 2025  
> Data license: CC-BY  
> Original source: https://ourworldindata.org/grapher/life-expectancy

## License

This project is released under the **MIT License** – feel free to use, modify, and share.

## Author

[Salvatore Spagnuolo] 

