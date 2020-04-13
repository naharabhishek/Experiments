loadNamespace("forecast")

handleForecast <- function(model, params) {
    outputs = list()
    output_types = "mean"
	
    if ("samples" %in% output_types) {
        outputs$samples <- lapply(1:params$num_samples, function(n) { simulate(model, params$prediction_length) } )
    }
    if("quantiles" %in% output_types) {
        f_matrix <- forecast::forecast(model, h=params$prediction_length, level=unlist(params$levels))$upper
        outputs$quantiles <- split(f_matrix, col(f_matrix))
    }
    if("mean" %in% output_types) {
        outputs$mean <- forecast::forecast(model, h=params$prediction_length)$mean
    }
    outputs
}


arima <- function(ts) {
    model <- forecast::auto.arima(ts, trace=TRUE)
}

predict <- function(model, params){
	outputs = list()
    output_types = "mean"
    output_types = params$type

    if ("samples" %in% output_types) {
        outputs$samples <- lapply(1:params$num_samples, function(n) { simulate(model, params$prediction_length) } )
    }
    if("quantiles" %in% output_types) {
        model_predict <- forecast::forecast(model, h=params$prediction_length, level=params$level)
        f_matrix <- model_predict$upper
        outputs$upper <- split(f_matrix, col(f_matrix))

        f_matrix <- model_predict$lower
        outputs$lower <- split(f_matrix, col(f_matrix))

        outputs$mean <- model_predict$mean
    }
    if("mean" %in% output_types) {
        outputs$mean <- forecast::forecast(model, h=params$prediction_length)$mean
    }
    outputs
}

ets <- function(ts) {
    model <- forecast::ets(ts, additive.only=TRUE)
}

croston <- function(ts, h) {
    model <- forecast::croston(ts, h=h)
}

tbats <- function(ts) {
    model <- forecast::tbats(ts)
}

mlp <- function(ts, params) {
    model <- nnfor::mlp(ts, hd.auto.type="valid")
}

naive <- function(ts, h, level) {
    model <- forecast::naive(y=ts, h=h, level=level)
}

snaive <- function(ts, h, level) {
    model <- forecast::snaive(y=ts, h=h, level=level)
}
