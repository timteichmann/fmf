target=$1
target_split=(${target//./ })
target=${target_split[0]}

gcc -shared -fPIC -O3 -o $target.so $target.c
