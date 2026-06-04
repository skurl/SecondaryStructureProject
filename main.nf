// // enabling nextflow DSL v2
// nextflow.enable.dsl = 2

// params.inputFile = "data/data.csv"
// params.resultsDir = "results"

// process PretrainSecondaryStructureModel {

//     publishDir "${params.resultsDir}", mode: 'copy'

//     container 'ghcr.io/stracquadaniolab/secondary-structure-project:0.0.1'

//     input:
//         path data_csv

//     output:
//         path 'outputs/*', optional: true

//     script:
//     """
//     mkdir -p data outputs
//     cp ${data_csv} data/data.csv
//     export OUTPUT_DIR=outputs
//     python bin/main.py
//     """
// }

// workflow {
//     data_ch = Channel.fromPath(params.inputFile, checkIfExists: true)
//     PretrainSecondaryStructureModel(data_ch)
// }

// enabling nextflow DSL v2

nextflow.enable.dsl = 2

params.inputFile = "data/data.csv"
params.resultsDir = "results"

process PretrainSecondaryStructureModel {

    publishDir "${params.resultsDir}", mode: 'copy'

    container './secondary-structure-project.sif'

    input:
        path data_csv

    output:
        path 'outputs/*', optional: true

    script:
    """
    mkdir -p data outputs
    cp ${data_csv} data/data.csv
    export OUTPUT_DIR=outputs
    python /app/bin/main.py
    """
}

workflow {
    data_ch = Channel.fromPath(params.inputFile, checkIfExists: true)
    PretrainSecondaryStructureModel(data_ch)
}
