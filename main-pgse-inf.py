from src.pipeline.pgse_inference_pipeline import Pipeline

MODEL_PATH = '../volatile/var/result-k6-CAZ-perf_fold_0.json'
SEGMENT_PATH = '../volatile/var/result-k6-CAZ-perf_fold_0.txt'

if __name__ == "__main__":
    # Instantiate the pipeline and keep it
    pipeline = Pipeline(MODEL_PATH, SEGMENT_PATH)

    # every time you want to run the pipeline to predict, just call the run method with
    # files as a list of paths to the fasta files
    EG_1 = [
        '../volatile/cgr/Sample_002-MOLMIC_B2.scaffolds.fna',
        '../volatile/cgr/Sample_394-MOLMIC_H8.scaffolds.fna',
        '../volatile/cgr/Sample_385-MOLMIC_G79.scaffolds.fna',
        '../volatile/cgr/Sample_622-MOLMIC_K68.scaffolds.fna',
        '../volatile/cgr/Sample_252-MOLMIC_F2.scaffolds.fna',
        '../volatile/cgr/Sample_208-MOLMIC_E33.scaffolds.fna',
        '../volatile/cgr/Sample_443-MOLMIC_H62.scaffolds.fna',
        '../volatile/cgr/Sample_565-MOLMIC_J66.scaffolds.fna',
        '../volatile/cgr/Sample_339-MOLMIC_G29.scaffolds.fna',
        '../volatile/cgr/Sample_418-MOLMIC_H33.scaffolds.fna',
    ]

    result_1 = pipeline.run(EG_1)
    print(result_1)

    EG_2 = [
        '../volatile/cgr/Sample_394-MOLMIC_H8.scaffolds.fna',
        '../volatile/cgr/Sample_385-MOLMIC_G79.scaffolds.fna',
        '../volatile/cgr/Sample_622-MOLMIC_K68.scaffolds.fna',
        '../volatile/cgr/Sample_252-MOLMIC_F2.scaffolds.fna'
    ]

    result_2 = pipeline.run(EG_2)
    print(result_2)