#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

(
    python main.py -m train --data amr -s -e 10 -l 2e-5 -b 16 -p
) &
wait $!
(
    python main.py -m train --data amr -s -e 10 -l 2e-5 -b 32 -p
) &
wait $!
(
    python main.py -m train --data amr -s -e 5 -l 2e-5 -b 16 -p
) &
wait $!
(
    python main.py -m train --data amr -s -e 5 -l 2e-5 -b 32 -p
) &
wait $!
(
    python main.py -m train --data amr -s -e 3 -l 2e-5 -b 16 -p
) &
wait $!
(
    python main.py -m train --data amr -s -e 3 -l 2e-5 -b 32 -p
) &

