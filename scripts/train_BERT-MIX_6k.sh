#!/bin/bash

export CUDA_VISIBLE_DEVICES=2

(
    python main.py -m train --data mix --mix 6 --mixid 1 -e 10 -l 2e-5 -b 32 -p
) &
wait $!
(
    python main.py -m train --data mix --mix 6 --mixid 1 -e 10 -l 3e-5 -b 32 -p
) &
wait $!
(
    python main.py -m train --data mix --mix 6 --mixid 1 -e 10 -l 5e-5 -b 32 -p
) &
wait $!
(
    python main.py -m train --data mix --mix 6 --mixid 1 -e 10 -l 2e-5 -b 16 -p
) &
wait $!
(
    python main.py -m train --data mix --mix 6 --mixid 1 -e 10 -l 3e-5 -b 16 -p
) &
wait $!
(
    python main.py -m train --data mix --mix 6 --mixid 1 -e 10 -l 5e-5 -b 16 -p
) &