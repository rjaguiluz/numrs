#' Get NumRs Version
#' @export
nr_version <- function() {
  .Call("numrs_version")
}

#' Print Startup Log
#' @export
nr_print_log <- function() {
  .Call("numrs_print_startup_log")
  invisible()
}
