from __future__ import division
import platform
import logging
import os
import time
import datetime
import calendar
import traceback

import numpy as np
import subprocess
import shutil
import zipfile
import gzip
# from dipy.align.reslice import reslice
from ..Processing.brain_processing import *


class ANTsRegistration:
    """
    Class for all registration-based processes, using ANTs as a backend.
    By default the python implementation is used because easily deployable. The c++ implementation can be used if a
    locally compiled/installed ANTs is available (must be manually specified).
    """
    def __init__(self):
        self.ants_reg_dir = ResourcesConfiguration.getInstance().ants_reg_dir
        self.ants_apply_dir = ResourcesConfiguration.getInstance().ants_apply_dir
        self.registration_folder = os.path.join(ResourcesConfiguration.getInstance().output_folder, 'registration/')
        os.makedirs(self.registration_folder, exist_ok=True)
        self.reg_transform = {}
        self.transform_names = []
        self.inverse_transform_names = []
        self.registration_computed = False
        self.backend = ResourcesConfiguration.getInstance().system_ants_backend

    def clear_cache(self):
        # In Python, registration files are stored in the temporary folder and must be removed.
        if self.backend == 'python':
            if 'fwdtransforms' in self.reg_transform.keys():
                for elem in self.reg_transform['fwdtransforms']:
                    if os.path.exists(elem):
                        os.remove(elem)
            if 'invtransforms' in self.reg_transform.keys():
                for elem in self.reg_transform['invtransforms']:
                    if os.path.exists(elem):
                        os.remove(elem)

        if os.path.exists(self.registration_folder):
            shutil.rmtree(self.registration_folder)

    def clear_output_folder(self):
        if os.path.exists(self.registration_folder):
            shutil.rmtree(self.registration_folder)

    def dump_and_clean(self):
        """
        Save all resources coming from the MNI space, might not be necessary.
        """
        try:
            # @FIXME. Only needed when having the c++ backend?
            # shutil.copyfile(src=os.path.join(self.registration_folder, 'Warped.nii.gz'),
            #                 dst=os.path.join(ResourcesConfiguration.getInstance().output_folder, 'input_to_mni.nii.gz'))
            # shutil.copyfile(src=os.path.join(self.registration_folder, 'input_tumor_mask_reg_atlas.nii.gz'),
            #                 dst=os.path.join(ResourcesConfiguration.getInstance().output_folder, 'input_tumor_to_mni.nii.gz'))

            shutil.copyfile(src=ResourcesConfiguration.getInstance().mni_atlas_brain_mask_filepath,
                            dst=os.path.join(self.registration_folder, 'brain_mask_to_MNI.nii.gz'))

            cortical_structures_folder = os.path.join(self.registration_folder, 'Cortical-structures')
            os.makedirs(cortical_structures_folder, exist_ok=True)

            shutil.copyfile(src=ResourcesConfiguration.getInstance().cortical_structures['MNI']['MNI']['Mask'],
                            dst=os.path.join(cortical_structures_folder, 'MNI_cortical_structures_mask_mni.nii.gz'))
            shutil.copyfile(
                src=ResourcesConfiguration.getInstance().cortical_structures['MNI']['Harvard-Oxford']['Mask'],
                dst=os.path.join(cortical_structures_folder, 'Harvard-Oxford_cortical_structures_mask_mni.nii.gz'))
            shutil.copyfile(src=ResourcesConfiguration.getInstance().cortical_structures['MNI']['Schaefer17']['Mask'],
                            dst=os.path.join(cortical_structures_folder,
                                             'Schaefer17_cortical_structures_mask_mni.nii.gz'))
            shutil.copyfile(src=ResourcesConfiguration.getInstance().cortical_structures['MNI']['Schaefer7']['Mask'],
                            dst=os.path.join(cortical_structures_folder,
                                             'Schaefer7_cortical_structures_mask_mni.nii.gz'))
            subcortical_structures_folder = os.path.join(self.registration_folder, 'Subcortical-structures')
            os.makedirs(subcortical_structures_folder, exist_ok=True)
            shutil.copyfile(src=ResourcesConfiguration.getInstance().subcortical_structures['MNI']['BCB']['Mask'],
                            dst=os.path.join(subcortical_structures_folder,
                                             'BCB_subcortical_structures_mask_mni.nii.gz'))
            if not ResourcesConfiguration.getInstance().diagnosis_full_trace:
                # If the transform files are save, could be removed here.
                pass
                # if os.path.exists(self.registration_folder):
                #     shutil.rmtree(self.registration_folder)
        except Exception as e:
            raise NameError("Impossible to perform cleaning and dumping in the ANTs registration instance with: {}".format(e))

    def compute_registration(self, moving: str, fixed: str, registration_method: str) -> None:
        """

        Compute the registration between the moving and fixed images according to the specified registration method.
        Given the specified backend in ResourcesConfiguration, either the python or cpp library will be used
        (the default ants backend runs in python).

        Parameters
        ----------
        moving : str
            Filepath of the moving image (to register).
        fixed : str
            Filepath of the fixed image (to register to).
        registration_method : str
            ANTs tag to specify which registration method to use (e.g., SyN).
        Returns
        -------
        None
        """
        if len(self.transform_names) != 0 and len(self.inverse_transform_names) != 0:
            return
        os.makedirs(self.registration_folder, exist_ok=True)
        try:
            if self.backend == 'python':
                self.compute_registration_python(moving, fixed, registration_method)
            elif self.backend == 'cpp':
                self.compute_registration_cpp(moving, fixed, registration_method)
        except Exception as e:
            raise RuntimeError(e)
        self.registration_computed = True
        return

    def compute_registration_cpp(self, moving, fixed, registration_method):
        logging.debug("Starting registration for patient.")

        if registration_method == 'SyN':
            registration_method = 'sq'

        if registration_method == 'sq':
            script_path = os.path.join(self.ants_reg_dir, 'antsRegistrationSyNQuick.sh')
            registration_method = 's'
        else:
            script_path = os.path.join(self.ants_reg_dir, 'antsRegistrationSyN.sh')

        try:
            if platform.system() == 'Windows':
                subprocess.call(["{script}".format(script=script_path),
                                 '-d{dim}'.format(dim=3),
                                 '-f{fixed}'.format(fixed=fixed),
                                 '-m{moving}'.format(moving=moving),
                                 '-o{output}'.format(output=self.registration_folder),
                                 '-t{trans}'.format(trans=registration_method),
                                 '-n{cores}'.format(cores=8),
                                 #'-p{precision}'.format(precision='f')
                                 # '-x[{mask_fixed},{mask_moving}]'.format(
                                 #     mask_fixed='',
                                 #     mask_moving='')])
                                 #'-x{mask}'.format(mask=fixed_mask_filepath)
                                 ], shell=True)
            else:
                subprocess.call(["{script}".format(script=script_path),
                                 '-d{dim}'.format(dim=3),
                                 '-f{fixed}'.format(fixed=fixed),
                                 '-m{moving}'.format(moving=moving),
                                 '-o{output}'.format(output=self.registration_folder),
                                 '-t{trans}'.format(trans=registration_method),
                                 '-n{cores}'.format(cores=8),
                                 #'-p{precision}'.format(precision='f')
                                 # '-x[{mask_fixed},{mask_moving}]'.format(
                                 #     mask_fixed='',
                                 #     mask_moving='')])
                                 #'-x{mask}'.format(mask=fixed_mask_filepath)
                                 ])

            if registration_method == 's':
                self.reg_transform['fwdtransforms'] = [os.path.join(self.registration_folder, '1Warp.nii.gz'),
                                                       os.path.join(self.registration_folder, '0GenericAffine.mat')]
                self.reg_transform['invtransforms'] = [os.path.join(self.registration_folder, '1InverseWarp.nii.gz'),
                                                       os.path.join(self.registration_folder, '0GenericAffine.mat')]
                self.transform_names = ['1Warp.nii.gz', '0GenericAffine.mat']
                self.inverse_transform_names = ['1InverseWarp.nii.gz', '0GenericAffine.mat']
        except Exception as e:
            raise RuntimeError('Cpp-based ANTs registration failed with: {}'.format(e))

    def compute_registration_python(self, moving, fixed, registration_method) -> None:
        """
        @FIXME: "antsRegistrationSyNQuick[s]" does not work across all platforms, so swapped with "SyN".
        Read docs for supported transforms: https://antspy.readthedocs.io/en/latest/_modules/ants/registration/interface.html
        """
        import ants
        try:
            logging.info("starting python-based ANTs registration with method: {}.".format(registration_method))
            moving_ants = ants.image_read(moving, dimension=3)
            fixed_ants = ants.image_read(fixed, dimension=3)
            if registration_method == 'antsRegistrationSyNQuick[s]' or registration_method == 'antsRegistrationSyN[s]':
                registration_method = 'SyN'

            self.reg_transform = ants.registration(fixed_ants, moving_ants, registration_method)
            warped_input = ants.apply_transforms(fixed=fixed_ants,
                                                  moving=moving_ants,
                                                  transformlist=self.reg_transform['fwdtransforms'],
                                                  interpolator='linear',
                                                  whichtoinvert=[False, False])
            warped_input_filename = os.path.join(self.registration_folder, 'input_volume_to_MNI.nii.gz')
            ants.image_write(warped_input, warped_input_filename)
        except Exception as e:
            raise RuntimeError('Python-based ANTs registration failed with: {}'.format(e))

    def apply_registration_transform(self, moving, fixed, interpolation='nearestNeighbor'):
        os.makedirs(self.registration_folder, exist_ok=True)
        try:
            if self.backend == 'python':
                return self.apply_registration_transform_python(moving, fixed, interpolation)
            elif self.backend == 'cpp':
                return self.apply_registration_transform_cpp(moving, fixed, interpolation)
        except Exception as e:
            raise RuntimeError('{}'.format(e))

    def apply_registration_transform_cpp(self, moving, fixed, interpolation='NearestNeighbor'):
        """
        Apply a registration transform onto the corresponding moving image.
        """
        optimization_method = 'Linear' if interpolation == 'linear' else 'NearestNeighbor'
        logging.debug("Apply registration transform to input volume: {}.".format(moving))
        script_path = os.path.join(self.ants_apply_dir, 'antsApplyTransforms')

        # transform_filenames = [os.path.join(self.registration_folder, x) for x in self.transform_names]
        transform_filenames = self.reg_transform['fwdtransforms']
        moving_registered_filename = os.path.join(self.registration_folder,
                                                  os.path.basename(moving).split('.')[0] + '_reg_atlas.nii.gz')

        if len(transform_filenames) == 4:
            args = ("{script}".format(script=script_path),
                    "-d", "3",
                    '-r', '{fixed}'.format(fixed=fixed),
                    '-i', '{moving}'.format(moving=moving),
                    '-t', '{transform}'.format(transform=transform_filenames[0]),
                    '-t', '{transform}'.format(transform=transform_filenames[1]),
                    '-t', '{transform}'.format(transform=transform_filenames[2]),
                    '-t', '{transform}'.format(transform=transform_filenames[3]),
                    '-o', '{output}'.format(output=moving_registered_filename),
                    '-n', '{type}'.format(type=optimization_method))
        elif len(transform_filenames) == 2:
            args = ("{script}".format(script=script_path),
                    "-d", "3",
                    '-r', '{fixed}'.format(fixed=fixed),
                    '-i', '{moving}'.format(moving=moving),
                    '-t', '{transform}'.format(transform=transform_filenames[0]),
                    '-t', '{transform}'.format(transform=transform_filenames[1]),
                    '-o', '{output}'.format(output=moving_registered_filename),
                    '-n', '{type}'.format(type=optimization_method))
        elif len(transform_filenames) == 1:
            args = ("{script}".format(script=script_path),
                    "-d", "3",
                    '-r', '{fixed}'.format(fixed=fixed),
                    '-i', '{moving}'.format(moving=moving),
                    '-t', '{transform}'.format(transform=transform_filenames[0]),
                    '-o', '{output}'.format(output=moving_registered_filename),
                    '-n', '{type}'.format(type=optimization_method))
        elif len(transform_filenames) == 0:
            raise IndexError('List of transforms is empty.')

        # Or just:
        # args = "bin/bar -c somefile.xml -d text.txt -r aString -f anotherString".split()
        try:
            if platform.system() == 'Windows':
                popen = subprocess.Popen(args, stdout=subprocess.PIPE, shell=True)
            else:
                popen = subprocess.Popen(args, stdout=subprocess.PIPE)
            popen.wait()
            output = popen.stdout.read()
            return moving_registered_filename
        except Exception as e:
            raise RuntimeError('Cpp-based ANTs apply registration failed with: {}'.format(e))

    def apply_registration_transform_python(self, moving: str, fixed: str, interpolation: str = 'nearestNeighbor') -> str:
        import ants
        try:
            moving_ants = ants.image_read(moving, dimension=3)
            fixed_ants = ants.image_read(fixed, dimension=3)
            warped_input = ants.apply_transforms(fixed=fixed_ants,
                                                 moving=moving_ants,
                                                 transformlist=self.reg_transform['fwdtransforms'],
                                                 interpolator=interpolation,
                                                 whichtoinvert=[False, False])
            warped_input_filename = os.path.join(self.registration_folder, 'warped_input_to_output_space.nii.gz')
            ants.image_write(warped_input, warped_input_filename)
            return warped_input_filename
        except Exception as e:
            raise RuntimeError('Python-based ANTs apply registration failed with: {}'.format(e))

    def apply_registration_inverse_transform(self, moving, fixed, interpolation='nearestNeighbor', label=''):
        os.makedirs(self.registration_folder, exist_ok=True)
        try:
            if self.backend == 'python':
                return self.apply_registration_inverse_transform_python(moving, fixed, interpolation, label)
            elif self.backend == 'cpp':
                return self.apply_registration_inverse_transform_cpp(moving, fixed, interpolation, label)
        except Exception as e:
            raise RuntimeError('{}'.format(e))

    def apply_registration_inverse_transform_cpp(self, moving, fixed, interpolation='NearestNeighbor', label=''):
        """
        Apply an inverse registration transform onto the corresponding moving labels.
        """
        optimization_method = 'NearestNeighbor' if interpolation == 'nearestNeighbor' else 'Linear'
        logging.debug("Apply registration transform to input volume annotation: {}.".format(moving))
        script_path = os.path.join(self.ants_apply_dir, 'antsApplyTransforms')

        # transform_filenames = [os.path.join(self.registration_folder, x) for x in self.inverse_transform_names]
        transform_filenames = self.reg_transform['invtransforms']
        moving_registered_filename = os.path.join(self.registration_folder, label + '_mask_to_input.nii.gz')
        os.makedirs(os.path.dirname(moving_registered_filename), exist_ok=True)

        if len(transform_filenames) == 4:  # Combined case?
            args = ("{script}".format(script=script_path),
                    "-d", "3",
                    '-r', '{fixed}'.format(fixed=fixed),
                    '-i', '{moving}'.format(moving=moving),
                    '-t', '{transform}'.format(transform=transform_filenames[0]),
                    '-t', '[{transform}, 1]'.format(transform=transform_filenames[1]),
                    '-t', '{transform}'.format(transform=transform_filenames[2]),
                    '-t', '[{transform}, 1]'.format(transform=transform_filenames[3]),
                    '-o', '{output}'.format(output=moving_registered_filename),
                    '-n', '{type}'.format(type=optimization_method))
        elif len(transform_filenames) == 2:  # SyN case with a .mat file and a .nii.gz file
            args = ("{script}".format(script=script_path),
                    "-d", "3",
                    '-r', '{fixed}'.format(fixed=fixed),
                    '-i', '{moving}'.format(moving=moving),
                    '-t', '{transform}'.format(transform=transform_filenames[0]),
                    '-t', '[{transform}, 1]'.format(transform=transform_filenames[1]),
                    '-o', '{output}'.format(output=moving_registered_filename),
                    '-n', '{type}'.format(type=optimization_method))
        elif len(transform_filenames) == 1:  # Rigid case with only an .mat file
            args = ("{script}".format(script=script_path),
                    "-d", "3",
                    '-r', '{fixed}'.format(fixed=fixed),
                    '-i', '{moving}'.format(moving=moving),
                    '-t', '[{transform}, 1]'.format(transform=transform_filenames[0]),
                    '-o', '{output}'.format(output=moving_registered_filename),
                    '-n', '{type}'.format(type=optimization_method))
        elif len(transform_filenames) == 0:
            raise IndexError('List of transforms is empty.')

        # Or just:
        # args = "bin/bar -c somefile.xml -d text.txt -r aString -f anotherString".split()
        try:
            if platform.system() == 'Windows':
                popen = subprocess.Popen(args, stdout=subprocess.PIPE, shell=True)
            else:
                popen = subprocess.Popen(args, stdout=subprocess.PIPE)
            popen.wait()
            output = popen.stdout.read()
            return moving_registered_filename
        except Exception as e:
            raise RuntimeError('Failed to apply inverse transforms on input image with: {}'.format(e))

    def apply_registration_inverse_transform_python(self, moving, fixed, interpolation='nearestNeighbor', label=''):
        import ants
        try:
            moving_ants = ants.image_read(moving, dimension=3)
            fixed_ants = ants.image_read(fixed, dimension=3)
            warped_input = ants.apply_transforms(fixed=fixed_ants,
                                                 moving=moving_ants,
                                                 transformlist=self.reg_transform['invtransforms'],
                                                 interpolator=interpolation,
                                                 whichtoinvert=[True, False])
            warped_input_filename = os.path.join(self.registration_folder, label + '_mask.nii.gz')
            # warped_input_filename = os.path.join(ResourcesConfiguration.getInstance().output_folder, 'patient',
            #                                           label + '_mask.nii.gz')
            os.makedirs(os.path.dirname(warped_input_filename), exist_ok=True)
            ants.image_write(warped_input, warped_input_filename)
            return warped_input_filename
        except Exception as e:
            raise RuntimeError('Failed to apply inverse transform with: {}'.format(e))
