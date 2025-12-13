# 01. NumRs Arrays (R)

## Class: `NumRsArray` (External Pointer)

## Creation & Memory Layout
**Important**: R matrices are **Column-Major**. NumRs is **Row-Major**.
Always transpose R matrices before converting if preserving spatial layout matters (e.g. for printing or specific weights).

```r
library(numrs)

# R Matrix (Col-Major)
m <- matrix(1:4, nrow=2)

# Convert to Row-Major vector for NumRs
row_major_vec <- as.numeric(t(m))

# Create NumRs Array
arr <- nr_array(row_major_vec, shape=c(2, 2))
```

## Inspection
```r
print(arr) # Prints shape and address
```
