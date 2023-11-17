# anlp-figlang

This repository contains resources for:
  1. [HW 3] replicating baseline results on MABL (Metaphors Across Borders and Languages), the multi-lingual figurative language inference dataset introduced in https://aclanthology.org/2023.findings-acl.525/.
  2. In future contain code for [HW 4] improving performance on MABL beyond the baseline results

We wrote our own code to replicate the results instead of relying on running the authors' code simply to build a solid foundation on which we can improvise in the hw4.

The models can be trained on babel using `src/babel_train.sh`. They can be tested using `src/babel_test.sh`. Code for reproducing the various analyses in the report, such as average word length, sentiment classification, and clustering can also be found in the `src` directory. The `few_shot_test_out` contains our results of few_shot training and the `xlmr_end_ckpt/test` and `xlmr_end_ckpt/translate_test` contains results for the zero-shot test and zero-shot translate test on fine-tuned XLMR large model.
