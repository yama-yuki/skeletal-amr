#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

(
    python main.py -m train --data amr -e 10 -l 5e-5 -b 16 -t BERT-WIKI/3_3e-05_64/0 -p
) &
wait $!
(
    python main.py -m train --data amr -e 10 -l 5e-5 -b 32 -t BERT-WIKI/3_3e-05_64/0 -p
) &
wait $!
(
    python main.py -m train --data amr -e 5 -l 5e-5 -b 16 -t BERT-WIKI/3_3e-05_64/0 -p
) &
wait $!
(
    python main.py -m train --data amr -e 5 -l 5e-5 -b 32 -t BERT-WIKI/3_3e-05_64/0 -p
) &
wait $!
(
    python main.py -m train --data amr -e 3 -l 5e-5 -b 16 -t BERT-WIKI/3_3e-05_64/0 -p
) &
wait $!
(
    python main.py -m train --data amr -e 3 -l 5e-5 -b 32 -t BERT-WIKI/3_3e-05_64/0 -p
) &
wait $!

