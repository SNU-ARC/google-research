#!/bin/bash
# Usage Example 1: ./trace_script.sh program=scann dataset=sift1m metric=dot_product L=800 m=64 threshold=0.55 w=800
# Usage Example 2: ./trace_script.sh program=faiss dataset=sift1m metric=dot_product L=800 m=64 k_star=256 w=800 is_gpu=true
for ARGUMENT in "$@"
do

    KEY=$(echo $ARGUMENT | cut -f1 -d=)
    VALUE=$(echo $ARGUMENT | cut -f2 -d=)

    case "$KEY" in
            program)           program=${VALUE} ;;
            dataset)           dataset=${VALUE} ;;
            metric)            metric=${VALUE} ;;
            L)                 L=${VALUE} ;;
            m)                 m=${VALUE} ;;
			threshold)         threshold=${VALUE} ;;
            w)                 w=${VALUE} ;;
			k_star)            k_star=${VALUE} ;;
			is_gpu)            is_gpu=${VALUE} ;;
            *)
    esac


done

echo "program = $program"
echo "dataset = $dataset"
echo "metric = $metric"
echo "L = $L"
echo "m = $m"
echo "threshold = $threshold"
echo "w = $w"
echo "k_star = $k_star"
echo "is_gpu = $is_gpu"

if [ "$program" == "scann" ]; then
	python3 split_and_run.py --program scann --dataset "$dataset" --metric "$metric" --L "$L" --m "$m" --threshold "$threshold" --w "$w" --num_split 1 --topk 1000 --reorder 1000 --eval_split > temp.out
	python3 make_trace.py --program scann --dataset "$dataset" --metric "$metric" --L "$L" --m "$m" --threshold "$threshold" --w "$w" --num_split 1 --topk 1000 --reorder 1000
elif [ "$program" == "faiss" ]; then
    if [ "$is_gpu" = "true" ]; then
        python3 split_and_run.py --program faiss --dataset "$dataset" --metric "$metric" --L "$L" --m "$m" --k_star "$k_star" --w "$w" --num_split 1 --topk 1000 --reorder 1000 --eval_split --is_gpu > temp.out
    else
        python3 split_and_run.py --program faiss --dataset "$dataset" --metric "$metric" --L "$L" --m "$m" --threshold "$threshold" --w "$w" --num_split 1 --topk 1000 --reorder 1000 --eval_split > temp.out
    fi
    python3 make_trace.py --program faiss --dataset "$dataset" --metric "$metric" --L "$L" --m "$m" --k_star "$k_star" --w "$w" --num_split 1 --topk 1000 --reorder 1000
else
	echo "Wrong or missing program argument!"
fi