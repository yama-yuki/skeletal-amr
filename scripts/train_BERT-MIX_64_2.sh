#!/bin/bash

export CUDA_VISIBLE_DEVICES=1

(
    python main.py -m train --data mix --mix 8 --mixid 1 -e 10 -l 2e-5 -b 64 -p
) &
wait $!
(
    python main.py -m train --data mix --mix 8 --mixid 1 -e 10 -l 3e-5 -b 64 -p
) &
wait $!
(
    python main.py -m train --data mix --mix 8 --mixid 1 -e 10 -l 5e-5 -b 64 -p
) &
wait $!
(
    python main.py -m train --data mix --mix 10 --mixid 1 -e 10 -l 2e-5 -b 64 -p
) &
wait $!
(
    python main.py -m train --data mix --mix 10 --mixid 1 -e 10 -l 3e-5 -b 64 -p
) &
wait $!
(
    python main.py -m train --data mix --mix 10 --mixid 1 -e 10 -l 5e-5 -b 64 -p
) &

