export CUDA_VISIBLE_DEVICES=""

for i in `seq 1 11`;
do
  echo worker $i
  python extract.py --save_dir ./vizdoom_skip3/${i} --num_generate 1000 &
  sleep 1.0
done
