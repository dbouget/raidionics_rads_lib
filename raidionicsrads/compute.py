from raidionicsrads.Utils.configuration_parser import ResourcesConfiguration
from raidionicsrads.NeuroDiagnosis.neuro_diagnostics import *
from raidionicsrads.MediastinumDiagnosis.mediastinum_diagnostics import *
import logging


def run_rads(config_filename: str) -> None:
    """

    """
    ResourcesConfiguration.getInstance().set_environment(config_path=config_filename)
    input_filename = ResourcesConfiguration.getInstance().input_volume_filename
    logging.info("Starting diagnosis for file: {}.\n".format(input_filename))
    start = time.time()
    diagnosis_task = ResourcesConfiguration.getInstance().diagnosis_task

    if diagnosis_task == 'neuro_diagnosis':
        runner = NeuroDiagnostics(input_filename=input_filename)
        runner.run()
    elif diagnosis_task == 'mediastinum_diagnosis':
        runner = MediastinumDiagnostics(input_filename=input_filename)
        runner.run()
    else:
        raise AttributeError('The provided diagnosis task {} is not supported yet.'.format(diagnosis_task))
    logging.info('Total time for generating the standardized report: {} seconds.\n'.format(time.time() - start))
