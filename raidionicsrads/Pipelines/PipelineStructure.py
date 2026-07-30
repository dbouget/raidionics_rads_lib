import json
import logging
import os.path
import time
import traceback
from copy import deepcopy
import concurrent.futures

from aenum import Enum, unique
from raidionicsseg.Utils.configuration_parser import ConfigResources

from ..Utils.utilities import get_type_from_string
from ..Utils.configuration_parser import ResourcesConfiguration
from .AbstractPipelineStep import AbstractPipelineStep
from .ClassificationStep import ClassificationStep
from .SegmentationStep import SegmentationStep
from .SegmentationRefinementStep import SegmentationRefinementStep
from .RegistrationStep import RegistrationStep
from .RegistrationDeployerStep import RegistrationDeployerStep
from .FeaturesComputationStep import FeaturesComputationStep
from .SurgicalReportingStep import SurgicalReportingStep
from .ModelSelectionStep import ModelSelectionStep
from .ReportingSelectionStep import ReportingSelectionStep


@unique
class TaskType(Enum):
    """

    """
    _init_ = 'value string'

    Class = 0, 'Classification'
    Seg = 1, 'Segmentation'
    Reg = 2, 'Registration'
    AReg = 3, "Apply registration"
    FComp = 4, "Features computation"
    SRep = 5, "Surgical reporting"
    SegRef = 6, "Segmentation refinement"
    ModSelec = 7, "Model selection"
    ReportSelec = 8, "Reporting selection"

    def __str__(self):
        return self.string


class Pipeline:
    """
    Class defining how an MRI volume should be handled.
    """
    _input_filepath = ""  # Full filepath to the current pipeline, stored in a json file
    _pipeline_json = {}  # Loaded pipeline from the aforementioned json file, stored as a dictionary
    _steps = {}  # Internal pipeline steps, inherited from AbstractPipelineStep, matching the steps inside the json dict.

    def __init__(self, input_filename: str) -> None:
        self.__reset()
        self._input_filepath = input_filename
        self.__init_from_scratch()

    def __reset(self):
        """
        All objects share class or static variables.
        An instance or non-static variables are different for different objects (every object has a copy).
        """
        self._input_filepath = ""
        self._pipeline_json = {}
        self._steps = {}

    def __init_from_scratch(self):
        """

        Returns
        -------

        """
        with open(self._input_filepath, 'r') as infile:
            self._pipeline_json = json.load(infile)

        self.__parse_pipeline_steps(pipeline=self._pipeline_json, initial=True)

    def __parse_pipeline_steps(self, pipeline: {}, initial: bool = True) -> None:
        self._steps = {}
        for i, s in enumerate(list(pipeline.keys())):
            task = get_type_from_string(TaskType, pipeline[s]["task"])
            step = None
            if task == TaskType.Class:
                step = ClassificationStep(pipeline[s])
                if pipeline[s]["target"][0] == "MRSequence" and not initial:
                    step.skip = True
            elif task == TaskType.Seg:
                step = SegmentationStep(pipeline[s])
            elif task == TaskType.SegRef:
                step = SegmentationRefinementStep(pipeline[s])
            elif task == TaskType.Reg:
                step = RegistrationStep(pipeline[s])
            elif task == TaskType.AReg:
                step = RegistrationDeployerStep(pipeline[s])
            elif task == TaskType.FComp:
                step = FeaturesComputationStep(pipeline[s])
            elif task == TaskType.SRep:
                step = SurgicalReportingStep(pipeline[s])
            elif task == TaskType.ModSelec:
                step = ModelSelectionStep(pipeline[s])
            elif task == TaskType.ReportSelec:
                step = ReportingSelectionStep(pipeline[s])
            if step:
                self._steps[str(i)] = step
            else:
                logging.warning(f"Step dismissed because task could not be matched.")

    def _group_steps_for_execution(self) -> list[list[tuple[str, AbstractPipelineStep]]]:
        """
        Partition the step list into sequential execution groups.
        Consecutive RegistrationStep instances are grouped into a single parallel batch, such as independent
        SegmentationStep.

        @TODO. Should add a global description for each group, for logging
        """
        groups: list[list[tuple[str, AbstractPipelineStep]]] = []
        reg_batch: list[tuple[str, AbstractPipelineStep]] = []
        reg_atlas_batch: list[tuple[str, AbstractPipelineStep]] = []
        areg_batch: list[tuple[str, AbstractPipelineStep]] = []
        areg_atlas_batch: list[tuple[str, AbstractPipelineStep]] = []
        seg_batch: list[tuple[str, AbstractPipelineStep]] = []
        seg_mul_batch: list[tuple[str, AbstractPipelineStep]] = []
        segref_batch: list[tuple[str, AbstractPipelineStep]] = []
        segref_mul_batch: list[tuple[str, AbstractPipelineStep]] = []
        segref_glob_batch: list[tuple[str, AbstractPipelineStep]] = []
        feat_comp_batch: list[tuple[str, AbstractPipelineStep]] = []

        for key in sorted(self._steps, key=int):
            step = self._steps[key]
            if isinstance(step, RegistrationStep):
                reg_batch.append((key, step))
                # if step.step_json["fixed"]["timestamp"] != -1:
                #     reg_batch.append((key, step))
                # else:
                #     reg_atlas_batch.append((key, step))
            elif isinstance(step, SegmentationStep):
                if len(step.step_json["inputs"]) == 1:
                    seg_batch.append((key, step))
                else:
                    seg_mul_batch.append((key, step))
            elif isinstance(step, SegmentationRefinementStep):
                if step.step_json["operation"] != "global_context" and len(self._steps[str(int(key)-1)].step_json["inputs"]) == 1:
                    segref_batch.append((key, step))
                elif step.step_json["operation"] != "global_context":
                    segref_mul_batch.append((key, step))
                else:
                    segref_glob_batch.append((key, step))
            elif isinstance(step, RegistrationDeployerStep):
                if step.step_json["fixed"]["timestamp"] != -1:
                    areg_batch.append((key, step))
                else:
                    areg_atlas_batch.append((key, step))
            elif isinstance(step, FeaturesComputationStep):
                feat_comp_batch.append((key, step))
            else:
                groups.append((key, step))

        final = []
        final.append([groups[0]])
        final.append(seg_batch)
        final.append(segref_batch)
        final.append(reg_batch)
        final.append(areg_batch)
        final.append(seg_mul_batch)
        final.append(segref_mul_batch)
        final.append(segref_glob_batch)
        # final.append(reg_batch)
        final.append(areg_atlas_batch)
        final.append(feat_comp_batch)
        final.extend([[x] for x in groups[1:]])

        return final

    def _deduplicate_batch(
            self, batch: list[tuple[str, AbstractPipelineStep]]
    ) -> list[tuple[str, AbstractPipelineStep]]:
        """
        Within a parallel batch, keep only the first occurrence of each unique step.
        Duplicates are logged and discarded before any setup or execution.
        """
        seen: dict[str, tuple[str, AbstractPipelineStep]] = {}
        for key, step in batch:
            sk = step.step_key()
            if sk not in seen:
                seen[sk] = (key, step)
            else:
                logging.debug(
                    "[PipelineStructure] Duplicate step discarded from batch: %s",
                    step.step_description
                )
        return list(seen.values())

    def setup(self, patient_parameters) -> None:
        """
        @TODO. Should not consider all classification tasks the same, the initial exception is only for the sequence
        classification which is mandatory for further disambiguation in model selection...
        @TODO. How to propagate down the probabilities/thresholding decision for the segmentation models (is it
        enough with the main_config.ini parameter?

        Parameters
        ----------
        patient_parameters

        Returns
        -------

        """
        logging.info('LOG: Pipeline setup - {} steps.'.format(len(self._steps)))
        final_pipeline = {}
        final_count = 0
        for s in list(self._steps.keys()):
            try:
                if self._steps[s].get_task() in [str(TaskType.Class), str(TaskType.ModSelec), str(TaskType.ReportSelec)]:
                    start = time.time()
                    logging.info("LOG: Pipeline - {desc} - Begin ({curr}/{tot})".format(
                        desc=self._steps[s].step_description,
                        curr=str(int(s) + 1),
                        tot=len(self._steps)))
                    try:
                        self._steps[s].setup(patient_parameters)
                    except Exception as e:
                        logging.warning("""[PipelineStructure] Setup phase of {} failed with:\n{}""".format(
                            self._steps[s].step_json, e))
                        logging.debug("Traceback: {}.".format(traceback.format_exc()))
                        continue
                    pipeline_backup = deepcopy(final_pipeline)
                    try:
                        if self._steps[s].get_task() == str(TaskType.Class):
                            patient_parameters = self._steps[s].execute()
                            final_count = final_count + 1
                            final_count_str = str(final_count)
                            final_pipeline[final_count_str] = {}
                            final_pipeline[final_count_str] = deepcopy(self._steps[s].step_json)
                        else:
                            task_optimal_pipeline = self._steps[s].execute()
                            for top in task_optimal_pipeline.keys():
                                final_count = final_count + 1
                                final_count_str = str(final_count)
                                final_pipeline[final_count_str] = {}
                                final_pipeline[final_count_str] = task_optimal_pipeline[top]
                    except Exception as e:
                        logging.warning("""[PipelineStructure] Execution phase of {} failed with:\n{}""".format(
                            self._steps[s].step_json, e))
                        logging.debug(f"Traceback: {traceback.format_exc()}.")
                        final_pipeline = deepcopy(pipeline_backup)
                        continue
                    logging.info('LOG: Pipeline - {desc} - Runtime: {time} seconds.'.format(
                        desc=self._steps[s].step_description,
                        time=time.time() - start))
                    logging.info("LOG: Pipeline - {desc} - End ({curr}/{tot})".format(
                        desc=self._steps[s].step_description,
                        curr=str(int(s) + 1),
                        tot=len(self._steps)))
                else:
                    final_count = final_count + 1
                    final_count_str = str(final_count)
                    final_pipeline[final_count_str] = {}
                    final_pipeline[final_count_str] = self._steps[s].step_json
            except Exception as e:
                if self._steps[s].inclusion == "required":
                    logging.error("""[Backend error] setup phase of {} failed with:\n{}""".format(
                        self._steps[s].step_json, e))
                    logging.debug("Traceback: {}.".format(traceback.format_exc()))
                    break
                else:
                    logging.warning("""[Backend warning] setup phase of {} failed with:\n{}""".format(
                        self._steps[s].step_json, e))
                    logging.debug("Traceback: {}.".format(traceback.format_exc()))
        self.__parse_pipeline_steps(pipeline=final_pipeline, initial=False)

        # Writing on disk the actual/final pipeline (for info and reuse in Raidionics)
        executed_pipeline_fn = os.path.join(ResourcesConfiguration.getInstance().output_folder, "executed_pipeline.json")
        with open(executed_pipeline_fn, 'w', newline='\n') as outfile:
            json.dump(final_pipeline, outfile, indent=4)
        return patient_parameters

    def execute(self, patient_parameters):
        logging.info('LOG: Pipeline - {} steps.'.format(len(self._steps)))
        for s in list(self._steps.keys()):
            start = time.time()
            logging.info("LOG: Pipeline - {desc} - Begin ({curr}/{tot})".format(desc=self._steps[s].step_description,
                                                                                curr=str(int(s) + 1),
                                                                                tot=len(self._steps)))
            try:
                self._steps[s].setup(patient_parameters)
            except Exception as e:
                if self._steps[s].inclusion == "required":
                    logging.error("""[Backend error] Setup phase of {} failed with:\n{}""".format(
                        self._steps[s].step_json, e))
                    logging.debug("Traceback: {}.".format(traceback.format_exc()))
                    break
                else:
                    logging.warning("""[Backend warning] Setup phase of {} failed with:\n{}""".format(
                        self._steps[s].step_json, e))
                    logging.debug("Traceback: {}.".format(traceback.format_exc()))
            try:
                patient_parameters = self._steps[s].execute()
            except Exception as e:
                if self._steps[s].inclusion == "required":
                    logging.error("""[Backend error] Execution phase of {} failed with:\n{}""".format(
                        self._steps[s].step_json, e))
                    logging.debug("Traceback: {}.".format(traceback.format_exc()))
                    break
                else:
                    logging.warning("""[Backend warning] Execution phase of {} failed with:\n{}""".format(
                        self._steps[s].step_json, e))
                    logging.debug("Traceback: {}.".format(traceback.format_exc()))
            logging.info('LOG: Pipeline - {desc} - Runtime: {time} seconds.'.format(desc=self._steps[s].step_description,
                                                                                    time=time.time() - start))
            logging.info("LOG: Pipeline - {desc} - End ({curr}/{tot})".format(desc=self._steps[s].step_description,
                                                                              curr=str(int(s) + 1),
                                                                              tot=len(self._steps)))
        return patient_parameters

    def _execute_step(self, key: str, step: AbstractPipelineStep, patient_parameters,
                      total: int) -> None:
        """
        Run a single step (setup + execute), modifying patient_parameters in place.
        """
        start = time.time()
        logging.info("LOG: Pipeline - {desc} - Begin ({curr}/{tot})".format(
            desc=step.step_description, curr=int(key) + 1, tot=total))
        try:
            step.setup(patient_parameters)
        except Exception as e:
            if step.inclusion == "required":
                raise
            logging.warning("[PipelineStructure] Setup of %s failed: %s", step.step_json, e)
            logging.debug("Traceback: %s", traceback.format_exc())
            return

        try:
            step.execute()
        except Exception as e:
            if step.inclusion == "required":
                raise
            logging.warning("[PipelineStructure] Execution of %s failed: %s", step.step_json, e)
            logging.debug("Traceback: %s", traceback.format_exc())

        logging.info("LOG: Pipeline - %s - Runtime: %.1fs", step.step_description, time.time() - start)

    def execute_parallel(self, patient_parameters):
        groups = self._group_steps_for_execution()
        total = len(self._steps)
        max_workers = ResourcesConfiguration.getInstance().num_workers

        logging.info('LOG: Pipeline - {} step groups.'.format(len(groups)))
        for i, group in enumerate(groups):
            start = time.time()
            logging.info("LOG: Pipeline - {desc} - Begin ({curr}/{tot})".format(desc="",
                                                                                curr=str(i + 1),
                                                                                tot=len(groups)))
            if len(group) == 1:
                key, step = group[0]
                self._execute_step(key, step, patient_parameters, total)
            else:
                # Parallel batch
                unique_group = self._deduplicate_batch(group)
                logging.info("LOG: Pipeline - Running %d steps in parallel.", len(group))
                # Setup all steps first (read-only from patient_parameters, safe without lock)
                for key, step in unique_group:
                    try:
                        step.setup(patient_parameters)
                    except Exception as e:
                        if step.inclusion == "required":
                            raise
                        logging.warning("[PipelineStructure] Setup of %s failed: %s", step.step_json, e)
                        step.skip = True

                # Execute in parallel
                with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
                    futures = {
                        pool.submit(step.execute): (key, step)
                        for key, step in unique_group if not step.skip
                    }
                    for future in concurrent.futures.as_completed(futures):
                        key, step = futures[future]
                        try:
                            future.result()
                        except Exception as e:
                            if step.inclusion == "required":
                                raise
                            logging.warning("[PipelineStructure] Execution %s failed: %s",
                                            step.step_json, e)
                logging.info(
                    'LOG: Pipeline - {desc} - Runtime: {time} seconds.'.format(desc="",
                                                                               time=time.time() - start))
                logging.info("LOG: Pipeline - {desc} - End ({curr}/{tot})".format(desc="",
                                                                                  curr=str(i + 1),
                                                                                  tot=len(groups)))
        return patient_parameters

    def cleanup(self):
        for s in list(self._steps.keys()):
            self._steps[s].cleanup()