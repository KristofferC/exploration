
Calls a material model written in julia from c++.

Output:
```
old plastic strain:
0.00194234, -0.000971171, -0.000971171, 0, 0, 0, 
new plastic strain:
0.00195295, -0.000976475, -0.000976475, 2.23902e-32, -4.96357e-33, -5.47616e-19, 
```

* Set Julia path in CMakeLists.txt (probably need a source build or paths are different)
* Precompilation seems to be broken with embedding so everything has to be precompiled first from the julia terminal.
