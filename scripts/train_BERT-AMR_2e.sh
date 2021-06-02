#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

(
    python main.py -m train --data amr -e 10 -l 2e-5 -b 16 -p
) &
wait $!
(
    python main.py -m train --data amr -e 10 -l 2e-5 -b 32 -p
) &
wait $!
(
    python main.py -m train --data amr -e 5 -l 2e-5 -b 16 -p
) &
wait $!
(
    python main.py -m train --data amr -e 5 -l 2e-5 -b 32 -p
) &
wait $!
(
    python main.py -m train --data amr -e 3 -l 2e-5 -b 16 -p
) &
wait $!
(
    python main.py -m train --data amr -e 3 -l 2e-5 -b 32 -p
) &

