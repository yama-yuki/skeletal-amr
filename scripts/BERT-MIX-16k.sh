#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

(
    python main.py -m train --data mix --mix 16 --mixid 1 -e 10 -l 5e-5 -b 16 -p
) &
wait $!
(
    python main.py -m train --data mix --mix 16 --mixid 2 -e 10 -l 5e-5 -b 16 -p
) &
wait $!
(
    python main.py -m train --data mix --mix 16 --mixid 3 -e 10 -l 5e-5 -b 16 -p
) &
wait $!
(
    python main.py -m train --data mix --mix 16 --mixid 4 -e 10 -l 5e-5 -b 16 -p
) &
wait $!
(
    python main.py -m train --data mix --mix 16 --mixid 5 -e 10 -l 5e-5 -b 16 -p
) &
