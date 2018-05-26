imgtxtenh -d 118.110 $1 png:- |
convert png:- -gravity center -extent 2274x342 \
-resize 758x114 \
-strip $1
