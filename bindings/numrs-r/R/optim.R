#' Create Dataset
#' @export
nr_dataset <- function(inputs, inputs_shape, targets, targets_shape, batch_size=32) {
  ptr <- .Call("numrs_dataset_new",
        as.double(inputs), as.integer(inputs_shape),
        as.double(targets), as.integer(targets_shape),
        as.integer(batch_size))
  structure(list(impl = ptr), class = "NumRsDataset")
}

#' Create Trainer Builder
#' @export
nr_trainer_builder <- function(model) {
  ptr <- .Call("numrs_trainer_builder_new", model$impl)
  structure(list(impl = ptr), class = "NumRsTrainerBuilder")
}

#' Set Learning Rate
#' @export
nr_with_lr <- function(builder, lr) {
  ptr <- .Call("numrs_trainer_builder_learning_rate", builder$impl, as.double(lr))
  structure(list(impl = ptr), class = "NumRsTrainerBuilder")
}

#' Build SGD Trainer (MSE Loss)
#' @export
nr_build_sgd <- function(builder) {
  ptr <- .Call("numrs_trainer_build_sgd_mse", builder$impl)
  structure(list(impl = ptr), class = "NumRsTrainer")
}

#' Fit Model
#' @export
nr_fit <- function(trainer, dataset, epochs=10) {
  .Call("numrs_trainer_fit_run", trainer$impl, dataset$impl, as.integer(epochs))
  # No return val yet, maybe future history
  invisible(NULL) 
}

#' Build Trainer (Generic)
#' @export
nr_build <- function(builder, optimizer="sgd", loss="mse") {
  ptr <- .Call("numrs_trainer_build", builder$impl, as.character(optimizer), as.character(loss))
  if (is.null(ptr)) stop(paste("Failed to build trainer with", optimizer, loss))
  structure(list(impl = ptr), class = "NumRsTrainer")
}
