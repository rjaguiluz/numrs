# 01. NumRs Arrays (C)

## Opaque Handle: `NumRsArray*`
In C, arrays are opaque pointers. You cannot access fields like `.shape` directly; you must use getter functions.

## Creation
Copy data from C pointers to NumRs managed memory.

```c
float data[] = {1.0, 2.0, 3.0, 4.0};
uint32_t shape[] = {2, 2};
// Creates a COPY of data.
NumRsArray *arr = numrs_array_new(data, shape, 2);
```

## Lifecycle
**Arrays must be manually freed.**

```c
numrs_array_free(arr);
```

## Inspection
```c
numrs_array_print(arr); // Prints to stdout
```
