#!/bin/bash

export CUDA_VISIBLE_DEVICES=2

(
    python main.py -m train --data amr -s -e 10 -l 5e-5 -b 16 -p
) &
wait $!
(
    python main.py -m train --data amr -s -e 10 -l 5e-5 -b 32 -p
) &
wait $!
(
    python main.py -m train --data amr -s -e 5 -l 5e-5 -b 16 -p
) &
wait $!
(
    python main.py -m train --data amr -s -e 5 -l 5e-5 -b 32 -p
) &
wait $!
(
    python main.py -m train --data amr -s -e 3 -l 5e-5 -b 16 -p
) &
wait $!
(
    python main.py -m train --data amr -s -e 3 -l 5e-5 -b 32 -p
) &

