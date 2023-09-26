from crisis_prediction.validators.compute_group_metrics import ComputeGroupMetrics


def validate_time_split_data(model, loader, config, preprocessors, target_name='crisis_in_4_weeks',
                             flag_average_precision=False):
    train_data = loader.load(start=config['train_start'], end=config['train_end'])
    test_data = loader.load(start=config['validation_start'], end=config['validation_end'])
    for preprocessor in preprocessors:
        train_data, test_data = preprocessor.preprocess(train_data, test_data)
    model.fit(train_data)

    test_data['predictions'] = model.predict(test_data)
    train_data['predictions'] = model.predict(train_data)
    compute_group_metrics = ComputeGroupMetrics([])
    test_metrics = compute_group_metrics.compute_binary_metrics(test_data[target_name], test_data['predictions'],
                                                                flag_average_precision)
    train_metrics = compute_group_metrics.compute_binary_metrics(train_data[target_name], train_data['predictions'],
                                                                 flag_average_precision)
    return train_metrics, test_metrics, train_data, test_data
