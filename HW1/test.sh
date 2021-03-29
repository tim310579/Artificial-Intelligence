#!/bin/bash

read -p "choose algorithm(1-5): " algo
echo `python test.py $algo prog1_puzzle/L01.txt` > "result/$algo/L01.txt"
echo `python test.py $algo prog1_puzzle/L02.txt` > "result/$algo/L02.txt"
echo `python test.py $algo prog1_puzzle/L03.txt` > "result/$algo/L03.txt"
echo `python test.py $algo prog1_puzzle/L04.txt` > "result/$algo/L04.txt"
echo `python test.py $algo prog1_puzzle/L10.txt` > "result/$algo/L10.txt"

echo `python test.py $algo prog1_puzzle/L11.txt` > "result/$algo/L11.txt"
echo `python test.py $algo prog1_puzzle/L20.txt` > "result/$algo/L20.txt"
echo `python test.py $algo prog1_puzzle/L21.txt` > "result/$algo/L21.txt"
echo `python test.py $algo prog1_puzzle/L22.txt` > "result/$algo/L22.txt"
echo `python test.py $algo prog1_puzzle/L23.txt` > "result/$algo/L23.txt"

echo `python test.py $algo prog1_puzzle/L24.txt` > "result/$algo/L24.txt"
echo `python test.py $algo prog1_puzzle/L25.txt` > "result/$algo/L25.txt"
echo `python test.py $algo prog1_puzzle/L26.txt` > "result/$algo/L26.txt"
echo `python test.py $algo prog1_puzzle/L27.txt` > "result/$algo/L27.txt"
echo `python test.py $algo prog1_puzzle/L28.txt` > "result/$algo/L28.txt"

echo `python test.py $algo prog1_puzzle/L29.txt` > "result/$algo/L29.txt"
echo `python test.py $algo prog1_puzzle/L30.txt` > "result/$algo/L30.txt"
echo `python test.py $algo prog1_puzzle/L31.txt` > "result/$algo/L31.txt"
echo `python test.py $algo prog1_puzzle/L40.txt` > "result/$algo/L40.txt"

