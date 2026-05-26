#!/bin/bash
gcc main.c -o solution

echo "--- Run Case 3: Empty Trees ---"
cat << 'IN' > case.in
0
5
10 -1
5 0
15 0
2 1
7 1
10
IN
./solution < case.in

echo -e "\n--- Run Case 4: Duplicates filtering ---"
cat << 'IN' > case.in
3
5 -1
5 0
5 0
3
5 -1
5 0
5 0
10
IN
./solution < case.in

echo -e "\n--- Run Case 6: Very large bound handling ---"
cat << 'IN' > case.in
2
1000000000 -1
1000000000 0
2
3000000000 -1
3000000000 0
4000000000
IN
./solution < case.in
