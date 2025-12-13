#' Create a NumRs Array
#' @param data Numeric vector
#' @param shape Integer vector of dimensions
#' @export
nr_array <- function(data, shape) {
  if (missing(shape)) shape <- dim(data)
  if (is.null(shape)) shape <- length(data)
  
  # Ensure data is valid double for C
  d_vec <- as.double(data)
  s_vec <- as.integer(shape)
  
  ptr <- .Call("numrs_array_new", d_vec, s_vec)
  
  # EAGER COPY: Create native R array for visualization
  # This ensures 'data' is visible in RStudio immediately
  # and creates a seamless integration experience.
  r_array <- d_vec
  dim(r_array) <- s_vec
  
  structure(list(
    impl = ptr,
    data = r_array,   # <--- The Magic Key for Visualization
    shape = s_vec,
    dtype = "Float32",
    device = "CPU"
  ), class = "NumRsArray")
}

#' Print NumRs Array
#' @export
print.NumRsArray <- function(x, ...) {
  # We can now just print the cached R data which matches R's native look
  cat("NumRsArray:\n")
  print(x$data)
}

#' Get Array Shape
#' @export
nr_shape <- function(x) {
  UseMethod("nr_shape")
}

#' @export
nr_shape.NumRsArray <- function(x) {
  # Robust check: if x is list (new format), check cache.
  if (is.list(x)) {
    if (!is.null(x$shape)) return(x$shape)
    ptr <- x$impl
  } else {
    # Legacy/Fallback: x is the pointer
    ptr <- x
  }
  .Call("numrs_array_shape", ptr)
}

#' @export
dim.NumRsArray <- function(x) {
  if (is.list(x)) {
    if (!is.null(x$shape)) return(x$shape)
    ptr <- x$impl
  } else {
    ptr <- x
  }
  .Call("numrs_array_shape", ptr)
}

#' Convert to R Array
#' @export
as.array.NumRsArray <- function(x, ...) {
  vec <- .Call("numrs_array_to_r", x$impl)
  dim(vec) <- dim(x)
  vec
}

#' Convert to Matrix
#' @export
as.matrix.NumRsArray <- function(x, ...) {
  as.matrix(as.array(x), ...)
}

#' Convert to Data Frame (for View())
#' @export
as.data.frame.NumRsArray <- function(x, ...) {
  as.data.frame(as.matrix(x), ...)
}

#' Convert to Vector
#' @export
as.double.NumRsArray <- function(x, ...) {
  .Call("numrs_array_to_r", x$impl)
}

#' Compact Structure Display
#' @export
str.NumRsArray <- function(object, ...) {
  s <- nr_shape(object)
  cat("NumRsArray of shape:", paste(s, collapse = "x"), "\n")
  # No need for extra details since it's a list now, RStudio handles inspection better
}
