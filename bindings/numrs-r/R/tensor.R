#' Create a Tensor from Array
#' @param array NumRsArray pointer
#' @param requires_grad Logical, track gradients?
#' @export
nr_tensor <- function(array, requires_grad = FALSE) {
  if (!inherits(array, "NumRsArray")) stop("Input must be a NumRsArray")
  ptr <- .Call("numrs_tensor_new", array$impl, as.logical(requires_grad))
  
  # Eagerly get data from the input array for visualization
  r_data <- if (!is.null(array$data)) array$data else .Call("numrs_array_to_r", array$impl)
  if (is.null(dim(r_data)) && !is.null(array$shape)) dim(r_data) <- array$shape

  structure(list(
    impl = ptr,
    data = r_data, # <--- Visible in RStudio!
    shape = array$shape,
    dtype = "Float32",
    requires_grad = requires_grad,
    device = "CPU"
  ), class = "NumRsTensor")
}

#' Get Tensor Data (as Array)
#' @export
nr_data <- function(tensor) {
  # If we already have cached data in tensor, wrapping it is fast
  if (!is.null(tensor$data)) {
    # But we need a new pointer for the Array view usually? 
    # Or just return the same impl? 
    # To be safe and correct with C API, we get the pointer.
    ptr <- .Call("numrs_tensor_data", tensor$impl)
    if (is.null(ptr)) return(NULL)
    
    return(structure(list(
      impl = ptr, 
      data = tensor$data, # Reuse cached data!
      shape = tensor$shape,
      dtype = "Float32",
      device = "CPU"
    ), class = "NumRsArray"))
  }

  ptr <- .Call("numrs_tensor_data", tensor$impl)
  if (is.null(ptr)) return(NULL)
  
  # Fallback query
  s <- .Call("numrs_tensor_shape", tensor$impl)
  
  # Fetch data eagerly for consistency with our new design
  val <- .Call("numrs_array_to_r", ptr)
  dim(val) <- s
  
  structure(list(
    impl = ptr, 
    data = val, # <--- Return cached data
    shape = s,
    dtype = "Float32",
    device = "CPU"
  ), class = "NumRsArray")
}

#' Get Tensor Gradient
#' @export
nr_grad <- function(tensor) {
  ptr <- .Call("numrs_tensor_grad", tensor$impl)
  if (is.null(ptr)) return(NULL)
  
  # Grad has same shape as tensor
  s <- if (!is.null(tensor$shape)) tensor$shape else .Call("numrs_tensor_shape", tensor$impl)
  
  structure(list(
    impl = ptr, 
    shape = s,
    dtype = "Float32",
    requires_grad = FALSE,
    device = "CPU"
  ), class = "NumRsTensor")
}

#' Perform Backward Pass
#' @export
nr_backward <- function(tensor) {
  .Call("numrs_tensor_backward", tensor$impl)
}

# Operators

# Helper to wrap result
wrap_tensor_result <- function(ptr) {
  # Get shape/data for visualization
  # We rely on previous S3 methods to convert
  # This might be circular if we use as.array(NumRsTensor) which uses nr_data().
  # We need a direct way to get data from the Tensor pointer.
  # The C function `r_numrs_array_to_r` works on ARRAY pointer.
  # Tensors wrap arrays.
  
  # To avoid complexity, we'll let print/str handle the lazy load?
  # No, user wants $data visible in list.
  
  # We need to get the inner array from the new tensor ptr to convert it.
  # But we don't have the inner array ptr exposed easily here without calling nr_data.
  # C-side: numrs_tensor_data(ptr) -> array_ptr -> r_numrs_array_to_r
  
  # Let's call the C methods directly to satisfy this
  # Ideally we'd expose a helper in C, but we can compose from R side
  
  # 1. Get inner array ptr
  arr_ptr <- .Call("numrs_tensor_data", ptr)
  
  # 2. Get data and shape
  s <- .Call("numrs_array_shape", arr_ptr)
  d <- .Call("numrs_array_to_r", arr_ptr)
  dim(d) <- s
  
  structure(list(
    impl = ptr,
    data = d,
    shape = s,
    dtype = "Float32", 
    device = "CPU",
    requires_grad = FALSE # Ops usually return non-leaf or intermediate, simplest assumption
  ), class = "NumRsTensor")
}

#' @export
nr_add <- function(a, b) {
  ptr <- .Call("numrs_add", a$impl, b$impl)
  wrap_tensor_result(ptr)
}

#' @export
nr_sub <- function(a, b) {
  ptr <- .Call("numrs_sub", a$impl, b$impl)
  wrap_tensor_result(ptr)
}

#' @export
nr_mul <- function(a, b) {
  ptr <- .Call("numrs_mul", a$impl, b$impl)
  wrap_tensor_result(ptr)
}

#' @export
nr_div <- function(a, b) {
  ptr <- .Call("numrs_div", a$impl, b$impl)
  wrap_tensor_result(ptr)
}

#' @export
nr_matmul <- function(a, b) {
  ptr <- .Call("numrs_matmul", a$impl, b$impl)
  wrap_tensor_result(ptr)
}

#' @export
nr_mse_loss <- function(pred, target) {
  ptr <- .Call("numrs_mse_loss", pred$impl, target$impl)
  s <- .Call("numrs_tensor_shape", ptr)
  structure(list(impl = ptr, shape=s, dtype="Float32", device="CPU"), class = "NumRsTensor")
}

# Activation
#' @export
nr_relu <- function(x) {
  ptr <- .Call("numrs_relu", x$impl)
  s <- if(!is.null(x$shape)) x$shape else .Call("numrs_tensor_shape", ptr)
  structure(list(impl = ptr, shape=s, dtype="Float32", device="CPU"), class = "NumRsTensor")
}

#' @export
nr_sigmoid <- function(x) {
  ptr <- .Call("numrs_sigmoid", x$impl)
  s <- if(!is.null(x$shape)) x$shape else .Call("numrs_tensor_shape", ptr)
  structure(list(impl = ptr, shape=s, dtype="Float32", device="CPU"), class = "NumRsTensor")
}

# S3 Methods

#' @export
`+.NumRsTensor` <- function(a, b) nr_add(a, b)

#' @export
`-.NumRsTensor` <- function(a, b) nr_sub(a, b)

#' @export
`*.NumRsTensor` <- function(a, b) nr_mul(a, b)

#' @export
`/.NumRsTensor` <- function(a, b) nr_div(a, b)

#' @export
`%*%` <- function(a, b) UseMethod("%*%")

#' @export
`%*%.NumRsTensor` <- function(a, b) nr_matmul(a, b)

#' @export
nr_shape.NumRsTensor <- function(x) {
  if (is.list(x)) {
    if (!is.null(x$shape)) return(x$shape)
    ptr <- x$impl
  } else {
    ptr <- x
  }
  .Call("numrs_tensor_shape", ptr)
}

#' @export
dim.NumRsTensor <- function(x) {
  if (is.list(x)) {
    if (!is.null(x$shape)) return(x$shape)
    ptr <- x$impl
  } else {
    ptr <- x
  }
  .Call("numrs_tensor_shape", ptr)
}

#' @export
as.array.NumRsTensor <- function(x, ...) {
  # Get data as NumRsArray first (creates copy/view)
  arr <- nr_data(x) 
  if (is.null(arr)) return(NULL)
  as.array(arr)
}

#' @export
as.matrix.NumRsTensor <- function(x, ...) {
  as.matrix(as.array(x), ...)
}

#' @export
as.data.frame.NumRsTensor <- function(x, ...) {
  as.data.frame(as.matrix(x), ...)
}

#' @export
as.double.NumRsTensor <- function(x, ...) {
  as.double(as.array(x), ...)
}

#' Print NumRs Tensor
#' @export
print.NumRsTensor <- function(x, ...) {
  cat("NumRsTensor:\n")
  if (!is.null(x$data)) {
    print(x$data)
  } else {
    # Lazy fetch for printing
    arr <- nr_data(x)
    if (!is.null(arr) && !is.null(arr$data)) {
      print(arr$data)
    } else {
      cat("<Unviewable Tensor>\n")
    }
  }
}

#' @export
str.NumRsTensor <- function(object, ...) {
  s <- nr_shape(object)
  cat("NumRsTensor of shape:", paste(s, collapse = "x"), "\n")
}

#' @export
nr_log <- function(a) {
  val <- .Call("numrs_log", a$impl)
  wrap_tensor_result(val)
}

#' @export
nr_mean <- function(a, axis=NULL) {
  # Wraps numrs_mean (which might reduce all)
  ptr <- .Call("numrs_mean", a$impl)
  s <- .Call("numrs_tensor_shape", ptr)
  # Result is generally a tensor
  structure(list(
    impl = ptr,
    shape = s,
    dtype = a$dtype, 
    device = a$device,
    requires_grad = TRUE,
    data = NULL
  ), class = "NumRsTensor")
}

#' @export
nr_reshape <- function(a, shape) {
  val <- .Call("numrs_reshape", a$impl, as.integer(shape))
  if (is.null(val)) stop("Reshape failed")
  
  new_shape <- .Call("numrs_tensor_shape", val)
  structure(list(
    impl = val,
    shape = new_shape,
    dtype = a$dtype,
    device = a$device,
    requires_grad = a$requires_grad,
    data = NULL
  ), class = "NumRsTensor")
}

#' @export
nr_flatten <- function(a, start_dim=0, end_dim=-1) {
  val <- .Call("numrs_flatten", a$impl, as.integer(start_dim), as.integer(end_dim))
  if (is.null(val)) stop("Flatten failed")
  
  new_shape <- .Call("numrs_tensor_shape", val)
  structure(list(
    impl = val,
    shape = new_shape,
    dtype = a$dtype,
    device = a$device,
    requires_grad = a$requires_grad,
    data = NULL
  ), class = "NumRsTensor")
}
