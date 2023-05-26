## Inference

* Dowload test dataset [HuggingFace SnakeCLEF2023](https://huggingface.co/spaces/competitions/SnakeCLEF2023) and keep at path `../../`
* Download Model [arti00/snakeCLEF2023-classification-model](https://huggingface.co/arti00/snakeCLEF2023-classification-model/tree/main)

| Model Arch              | Model Name                                                                  | Prediction File           |
| ----------------------- | ----------------------------------------------------------------------------| ------------------------- |
| deit_base_distilled_384 | clef2023_deit_base_distilled_384_efocal_05-12-2023_12-14-53.pth             | prediction.py /ipynb       |
| deit_base_distilled_384 | clef2023_deit_base_distilled_384_ensemble_focal_05-15-2023_12-27-11.pth     | prediction.py /ipynb       |
| efficientnet_b0         | clef2023_efficientnet_b0_focal_05-10-2023_02-33-37.pth                      | prediction.py /ipynb       |
|      -----              | ensemble-model-clef2023_vit_small_384_ensemble_focal_05-14-2023_02-18-03.pth| prediction_ensemble.ipynb |




* Keep model files in ../results/models/
* The prediction file will generate 2 `csv` files. `test_prediction.csv` has all duplicate `observation_id` while `snake_prediction.csv` has unique `observation_id` and is the final output file.

* run `python3 prediction.py --data_dir ../../ --model_arch <model_architectures> --model_name <filename_of_model> --data_csv <path_to_test_csv> --model_path <model_directory>` 

When running `prediction_ensemble.ipynb` change files paths. 



## Uniqueness of the Project

* The speciality of this project is development of ensemble loss function which uses Metric mentioned in the Competition as a weight for 
focal loss ensembled with equilized loss function. 

* Another different point is concatenation of outer layer is done in ensemble method, It can work better if trained more epoch with more 
resources.

* An experiment was done by building hierarchical method where firstly model is inferring 0 or 1 based one 2 classes i.e venom or not venom.
Then for next stage two models are trained separately trained, one on venomous lables with class_id as lables and another with non-venomous
labels with class_id as labels. If first stage predicts 0 then for next stage the prediction goes to the model trained on non-venom dataset
 and similarly for other label at first stage.
 
 To run hierarchical prediction run `prediction_hierarchical.ipynb` file. (Note: It is last minute experiment, models may not perform well)
 Models for `prediction_hierarchical.ipynb` are - 
 
 ```
models/non_v-clef2023_vit_small_384_ensemble_focal_05-22-2023_12-38-44.pth
models/venom-clef2023_vit_small_384_ensemble_focal_05-22-2023_05-09-10.pth
models/vornot-clef2023_vit_small_384_efocal_05-22-2023_16-25-34.pth
 ```
 
 
