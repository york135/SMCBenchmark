# SMCBenchmark

The source code to create the proposed SMC benchmark (unquantized, but all converted to npy formats).

The processed data is available at zenodo: [https://zenodo.org/records/15621863](https://zenodo.org/records/15621863)

For the data that has been converted to the format of MidiBERT (and our proposed [M2BERT](https://github.com/york135/M2BERT)),  see [https://github.com/york135/M2BERT_files](https://github.com/york135/M2BERT_files).

## Credit

| Task | Dataset                                               | Original source/repo                                                                                                                                  |
|:----:|:-----------------------------------------------------:|:----------------------------------------------------------------------------------------------------------------------------------------------------- |
| SGC  | Tagtraum (CD2, using the intersection w/ lmd_matched) | https://www.tagtraum.com/msd_genre_datasets.html                                                                                                      |
| BP   | Liu et al. (PM2S)                                     | See the description of https://github.com/cheriell/PM2S/tree/main/dev                                                                                 |
| DbP  | Liu et al. (PM2S)                                     | See the description of https://github.com/cheriell/PM2S/tree/main/dev                                                                                 |
| CR   | López et al. (AugmentedNet)                           | See the description of https://github.com/napulen/AugmentedNet                                                                                        |
| LK   | López et al. (AugmentedNet)                           | See the description of https://github.com/napulen/AugmentedNet                                                                                        |
| ME   | POP909                                                | https://github.com/music-x-lab/POP909-Dataset (processed with the code of MidiBERT, see https://github.com/wazenmai/MIDI-BERT/tree/CP/data_creation ) |
| VE   | POP909                                                | https://github.com/music-x-lab/POP909-Dataset (processed with the code of MidiBERT, see https://github.com/wazenmai/MIDI-BERT/tree/CP/data_creation ) |
| OTC  | Le et al.                                             | https://gitlab.com/algomus.fr/orchestration (using the pre-processed data from https://github.com/YaHsuanChu/orchestraTextureClassification )         |
| PS   | Pianist8                                              | https://zenodo.org/record/5089279 (processed with the code of MidiBERT, see https://github.com/wazenmai/MIDI-BERT/tree/CP/data_creation )             |
| ER   | EMOPIA                                                | https://annahung31.github.io/EMOPIA/ (processed with the code of MidiBERT, see https://github.com/wazenmai/MIDI-BERT/tree/CP/data_creation )          |
| VF   | TNUA                                                  | https://github.com/Tsung-Ping/Violin-Fingering-Generation/tree/master/TNUA_violin_fingering_dataset                                                   |
| MNID | BPS-motif                                             | https://github.com/Wiilly07/Beethoven_motif                                                                                                           |
