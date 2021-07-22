#!/bin/bash

dir="$1"
path="$2"
algo="$3"
num=$4

echo "Evaluating checkpoint $dir/$path"

script_name="../arch2vec/search_methods/$algo.py"
nb_dataset="../../info-nas/data/nb_dataset.json"

ls $dir

embedding_path="features_$path"

# extract features if not yet done
if [ ! -f "$dir/$embedding_path" ]; then
  echo "Extracting arch2vec..."
  python ../arch2vec/search_methods/reinforce.py --dim 16 --model_path $path --dir_name $dir --data_path $nb_dataset \
    --save_path $embedding_path
else
  echo "Features already extracted."
fi

if [ "$algo" = "reinforce" ]; then
  more_args="--bs 16 --saved_arch2vec"
else
  more_args="--init_size 16 --topk 5"
fi

if [ $num -ne 0 ]; then
  # run the search algorithm
  echo "Running search algorithm $algo."
  for s in $(seq 1 $num)
	  do
	    echo $s
	    run_fname=`echo -n $embedding_path | sed "s/.pt/.json/"`
	    echo "$dir""$algo"-runs/run_"$s"_"$run_fname"
	    if [ -f "$dir"/"$algo"-runs/run_"$s"_"$run_fname" ]; then
	      echo "Skipping"
	      continue
	    fi

      python $script_name --dim 16 --seed $s $more_args \
        --emb_path $embedding_path --dir_name $dir
    done
else
  echo "Finished feature extraction."
fi
