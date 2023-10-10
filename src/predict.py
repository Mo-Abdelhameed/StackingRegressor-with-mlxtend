import pandas as pd

from config import paths
from logger import get_logger
from Regressor import Regressor, predict_with_model
from schema.data_schema import load_saved_schema
from utils import read_csv_in_directory, save_dataframe_as_csv, read_json_as_dict
from preprocessing.pipeline import run_pipeline

logger = get_logger(task_name="predict")


def run_batch_predictions(
    test_dir=paths.TEST_DIR,
    predictor_dir=paths.PREDICTOR_DIR_PATH,
    predictions_file_path=paths.PREDICTIONS_FILE_PATH,
    saved_schema_dir=paths.SAVED_SCHEMA_DIR_PATH,
) -> None:
    """
    Run batch predictions on test data, save the predicted probabilities to a CSV file.

    This function reads test data from the specified directory,
    loads the preprocessing pipeline and pre-trained predictor model,
    transforms the test data using the pipeline,
    makes predictions using the trained predictor model,
    adds ids into the predictions dataframe,
    and saves the predictions as a CSV file.
    """
    model_config = read_json_as_dict(paths.MODEL_CONFIG_FILE_PATH)
    x_test = read_csv_in_directory(test_dir)
    data_schema = load_saved_schema(saved_schema_dir)
    ids = x_test[data_schema.id]
    x_test.drop(columns=data_schema.id, inplace=True)

    for column in data_schema.categorical_features:
        x_test[column] = x_test[column].astype(str)

    x_test = run_pipeline(x_test, data_schema, training=False)

    model = Regressor.load(predictor_dir)

    logger.info("Making predictions...")
    predictions_arr = predict_with_model(model, x_test)
    predictions_df = pd.DataFrame({data_schema.id: ids, model_config["prediction_field_name"]: predictions_arr})

    logger.info("Saving predictions...")
    save_dataframe_as_csv(dataframe=predictions_df, file_path=predictions_file_path)
    logger.info("Batch predictions completed successfully")


if __name__ == "__main__":
    run_batch_predictions()
