#/bin/bash

for n in $(seq 0.01 .01 0.7);
do
    python script.py $n
done

# for i in 0.64 0.8
# do
#     python script.py $i
# done
