# 03. Time Series Forecasting (R)

## 1D CNN Example
Source: `examples/timeseries_cnn.R`

```r
model <- nr_sequential(
    nr_conv1d(in_chan=1, out_chan=16, kernel=3, stride=1, padding=1),
    nr_relu_layer(),
    nr_flatten_layer(1, -1),
    nr_linear(16 * seq_len, 1)
)
```

## Data Input
Input tensor must be 3D.
```r
# R: [N, L] -> Transpose -> [N, 1, L] logic
# Remember dim setting
dim(arr) <- c(N, 1, L)
```
