gpu=0
for i in 0
do
  echo "${i}"
  python main.py --gpu $gpu --dataset anomaly_archive --seed $i
  # python main.py --gpu $gpu --dataset iops --seed $i
  # python main.py --gpu $gpu --dataset smd --seed $i
  # python main.py --gpu $gpu --dataset smap --seed $i
  # python main.py --gpu $gpu --dataset msl --seed $i
done