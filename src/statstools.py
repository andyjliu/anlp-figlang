#!/usr/bin/env python3
import sys
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, ttest_rel, pearsonr, spearmanr, zscore, norm
from statsmodels.stats.multitest import multipletests


# borrowed from https://github.com/maartensap/resources/blob/master/stats/statstools.py

def ttestSummaries(df,condition_col,measure_cols,paired=None):
  """Function that compares a set of features in two groups.
  df: data containing measures and conditions
  condition_col: column name containing the group belonging (e.g., control vs. treatment)
  measure_cols: column names to compare accross groups (e.g., num words, num pronouns, etc)
  paired: None if indep. t-test, else: name of column to pair measures on.
  """
  d = {}
  
  if paired:
    df = df.loc[~df[paired].isnull()]
    df = df.sort_values(by=[paired,condition_col])
    
  for m in measure_cols:
    d[m] = ttestSummary(df,condition_col,m,paired=paired)
    
  statDf = pd.DataFrame(d).T
  statDf["p_holm"] = multipletests(statDf["p"],method="h")[1]
  return statDf

def ttestSummary(df,condition_col,measure_col,paired=None):
  # conds = sorted(list(df[condition_col].unique()))
  conds = sorted(filter(lambda x: not pd.isnull(x),df[condition_col].unique()))

  conds = conds[:2]
  assert len(conds) == 2, "Not supported for more than 2 conditions "+str(conds)
  
  a = conds[0]
  b = conds[1]
  
  ix = ~df[measure_col].isnull()
  if paired:
    # merge and remove items that don't have two pairs
    pair_counts = df[ix].groupby(by=paired)[measure_col].count()
    pair_ids = pair_counts[pair_counts == 2].index
    ix = df[paired].isin(pair_ids)
    
  s_a = df.loc[(df[condition_col] == a) & ix,measure_col]
  s_b = df.loc[(df[condition_col] == b) & ix,measure_col]

  out = {
    f"mean_{a}": s_a.mean(),
    f"mean_{b}": s_b.mean(),
    f"std_{a}": s_a.std(),
    f"std_{b}": s_b.std(),
    f"n_{a}": len(s_a),
    f"n_{b}": len(s_b),    
  }
  if paired:    
    t, p = ttest_rel(s_a,s_b)
  else:
    t, p = ttest_ind(s_a,s_b)
    
  out["t"] = t
  out["p"] = p

  # Cohen's d  
  out["d"] = (s_a.mean() - s_b.mean()) / (np.sqrt(( s_a.std() ** 2 + s_b.std() ** 2) / 2))
  
  return out