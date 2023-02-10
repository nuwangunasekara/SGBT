# Gradient-Boosted-Trees-for-Evolving-Data-Streams

# Run MOA experiments with [DJL](https://djl.ai)
## Requirements
* [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)
## Datasets
### datasets from Streaming Random Patches (SRP): https://github.com/hmgomes/StreamingRandomPatches/tree/master/datasets
### synthetic data sets:
```
LED_a_S="-s (ConceptDriftStream -s (generators.LEDGeneratorDrift -d 1)   -d (ConceptDriftStream -s (generators.LEDGeneratorDrift -d 3) -d (ConceptDriftStream -s (generators.LEDGeneratorDrift -d 5)  -d (generators.LEDGeneratorDrift -d 7) -w 50 -p 250000 ) -w 50 -p 250000 ) -w 50 -p 250000 -r $random_seed )"
LED_g_S="-s (ConceptDriftStream -s (generators.LEDGeneratorDrift -d 1)   -d (ConceptDriftStream -s (generators.LEDGeneratorDrift -d 3) -d (ConceptDriftStream -s (generators.LEDGeneratorDrift -d 5)  -d (generators.LEDGeneratorDrift -d 7) -w 50000 -p 250000 ) -w 50000 -p 250000 ) -w 50000 -p 250000 -r $random_seed )"

AGR_a_S="-s (ConceptDriftStream -s (generators.AgrawalGenerator -f 1) -d (ConceptDriftStream -s (generators.AgrawalGenerator -f 2) -d (ConceptDriftStream -s (generators.AgrawalGenerator )   -d (generators.AgrawalGenerator -f 4) -w 50 -p 250000 ) -w 50 -p 250000 ) -w 50 -p 250000 -r $random_seed )"
AGR_g_S="-s (ConceptDriftStream -s (generators.AgrawalGenerator -f 1) -d (ConceptDriftStream -s (generators.AgrawalGenerator -f 2) -d (ConceptDriftStream -s (generators.AgrawalGenerator )   -d (generators.AgrawalGenerator -f 4) -w 50000 -p 250000 ) -w 50000 -p 250000 ) -w 50000 -p 250000 -r $random_seed )"

RBF_m_S="-s (generators.RandomRBFGeneratorDrift -c 5 -s .0001 -r $random_seed -i $random_seed)"
RBF_f_S="-s (generators.RandomRBFGeneratorDrift -c 5 -s .001 -r $random_seed -i $random_seed)"

RBF_Bm_S="-s (generators.RandomRBFGeneratorDrift -c 2 -s .0001 -r $random_seed -i $random_seed)"
RBF_Bf_S="-s (generators.RandomRBFGeneratorDrift -c 2 -s .001 -r $random_seed -i $random_seed)"

RandomTreeGenerator_S="-s (generators.RandomTreeGenerator -r $random_seed -i $random_seed)"

RandomRBF5_S="-s (generators.RandomRBFGenerator -r $random_seed -i $random_seed -c 5)"

LED_S="-s (generators.LEDGenerator -i $random_seed)"
```
## How to set up environment
###From source root run:
> bash ./moa/src/main/scripts/reinit_conda.sh ~/Desktop/conda/ moa/src/main/scripts/conda.yml
## How to build MOA for
###From source root run:
> bash ./moa/src/main/scripts/build_moa.sh ~/Desktop/m2/ ~/Desktop/conda/
## Run GUI
> bash ./moa/src/main/scripts/build_moa.sh ~/Desktop/m2/ ~/Desktop/conda/  ~/Desktop/djl.ai
## Run experiments
###From < results dir > run:
> bash < moa source root >/moa/src/main/scripts/run_moa.sh < dataset dir > < results dir > < DJL cache dir > < directory for separate maven repository > < directory for conda environment >

e.g
> bash ~/Desktop/moa_fork/moa/src/main/scripts/run_moa.sh ~/Desktop/datasets/ ~/Desktop/results/Exp1/ ~/Desktop/djl.ai/ ~/Desktop/m2/ ~/Desktop/conda/
