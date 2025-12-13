#' Create Sequential Model
#' @export
nr_sequential <- function(...) {
  layers <- list(...)
  seq <- .Call("numrs_sequential_new")
  s3_seq <- structure(list(impl = seq), class = "NumRsSequential")
  
  for (layer in layers) {
    if (inherits(layer, "NumRsLinear")) {
      .Call("numrs_sequential_add_linear", seq, layer$impl)
    } else if (inherits(layer, "NumRsReLU")) {
      .Call("numrs_sequential_add_relu", seq, layer$impl)
    } else if (inherits(layer, "NumRsConv1d")) {
      .Call("numrs_sequential_add_conv1d", seq, layer$impl)
    } else if (inherits(layer, "NumRsBatchNorm1d")) {
      .Call("numrs_sequential_add_batchnorm1d", seq, layer$impl)
    } else if (inherits(layer, "NumRsDropout")) {
      .Call("numrs_sequential_add_dropout", seq, layer$impl)
    } else if (inherits(layer, "NumRsFlatten")) {
      .Call("numrs_sequential_add_flatten", seq, layer$impl)
    } else if (inherits(layer, "NumRsSigmoid")) {
      .Call("numrs_sequential_add_sigmoid", seq, layer$impl)
    } else if (inherits(layer, "NumRsSoftmax")) {
      .Call("numrs_sequential_add_softmax", seq, layer$impl)
    } else {
      stop("Unsupported layer type in Sequential")
    }
  }
  s3_seq
}

#' Create Linear Layer
#' @export
nr_linear <- function(in_features, out_features) {
  ptr <- .Call("numrs_linear_new", as.integer(in_features), as.integer(out_features))
  structure(list(impl = ptr), class = "NumRsLinear")
}

#' Create ReLU Layer
#' @export
nr_relu_layer <- function() {
  ptr <- .Call("numrs_relu_layer_new")
  structure(list(impl = ptr), class = "NumRsReLU")
}

#' @export
nr_conv1d <- function(in_channels, out_channels, kernel_size, stride=1, padding=0) {
  ptr <- .Call("numrs_conv1d_new", as.integer(in_channels), as.integer(out_channels), 
               as.integer(kernel_size), as.integer(stride), as.integer(padding))
  structure(list(impl = ptr), class = "NumRsConv1d")
}

#' @export
nr_batchnorm1d <- function(num_features) {
  ptr <- .Call("numrs_batchnorm1d_new", as.integer(num_features))
  structure(list(impl = ptr), class = "NumRsBatchNorm1d")
}

#' @export
nr_dropout <- function(p=0.5) {
  ptr <- .Call("numrs_dropout_new", as.numeric(p))
  structure(list(impl = ptr), class = "NumRsDropout")
}

#' @export
nr_flatten_layer <- function(start_dim=1, end_dim=-1) {
  ptr <- .Call("numrs_flatten_layer_new", as.integer(start_dim), as.integer(end_dim))
  structure(list(impl = ptr), class = "NumRsFlatten")
}

#' @export
nr_sigmoid_layer <- function() {
  ptr <- .Call("numrs_sigmoid_layer_new")
  structure(list(impl = ptr), class = "NumRsSigmoid")
}

#' @export
nr_softmax_layer <- function() {
  ptr <- .Call("numrs_softmax_layer_new")
  structure(list(impl = ptr), class = "NumRsSoftmax")
}

#' Add Linear Layer to Sequential
#' @export
nr_add_linear <- function(seq, layer) {
  .Call("numrs_sequential_add_linear", seq$impl, layer$impl)
  invisible(seq)
}

#' Add ReLU Layer to Sequential
#' @export
nr_add_relu <- function(seq, layer) {
  .Call("numrs_sequential_add_relu", seq$impl, layer$impl)
  invisible(seq)
}

#' @export
nr_add_conv1d <- function(seq, layer) {
  .Call("numrs_sequential_add_conv1d", seq$impl, layer$impl)
  invisible(seq)
}

#' @export
nr_add_batchnorm1d <- function(seq, layer) {
  .Call("numrs_sequential_add_batchnorm1d", seq$impl, layer$impl)
  invisible(seq)
}

#' @export
nr_add_dropout <- function(seq, layer) {
  .Call("numrs_sequential_add_dropout", seq$impl, layer$impl)
  invisible(seq)
}

#' @export
nr_add_flatten <- function(seq, layer) {
  .Call("numrs_sequential_add_flatten", seq$impl, layer$impl)
  invisible(seq)
}

#' @export
nr_add_sigmoid <- function(seq, layer) {
  .Call("numrs_sequential_add_sigmoid", seq$impl, layer$impl)
  invisible(seq)
}

#' @export
nr_add_softmax <- function(seq, layer) {
  .Call("numrs_sequential_add_softmax", seq$impl, layer$impl)
  invisible(seq)
}

#' Forward Pass
#' @export
nr_forward <- function(model, input) {
  ptr <- .Call("numrs_sequential_forward", model$impl, input$impl)
  structure(list(impl = ptr), class = "NumRsTensor")
}
