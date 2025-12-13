# 06. Hands On: C API Example

## Simple Operations

```c
#include "numrs.h"

int main() {
    numrs_print_startup_log();
    
    float d[] = {1, 2, 3, 4};
    uint32_t s[] = {2, 2};
    
    NumRsArray *a = numrs_array_new(d, s, 2);
    NumRsTensor *t = numrs_tensor_new(a, false);
    
    NumRsTensor *res = numrs_add(t, t);
    
    // Cleanup
    numrs_tensor_free(t);
    numrs_tensor_free(res);
    numrs_array_free(a);
    return 0;
}
```

## Compilation
```bash
gcc main.c -lnumrs_c -L./lib -I./include
```
