random_seed=121
#random_seed=17
#random_seed=9

dir='/Users/ng98/Desktop/datasets/NEW/unzipped'

dataset=(elecNormNew nomao airlines covtypeNorm spam_corpus kdd99 WISDM_ar_v1.1_transformed)

for (( i=0; i<${#dataset[@]}; i++ ))
do
  in_file="${dir}/${dataset[$i]}.arff"
  out_file="${dir}/${dataset[$i]}RANDOM${random_seed}.arff"
  java -classpath /Users/ng98/Desktop/weka.jar weka.filters.unsupervised.instance.Randomize \
  -i "${in_file}" \
  -o "${out_file}" \
  -S ${random_seed}
done